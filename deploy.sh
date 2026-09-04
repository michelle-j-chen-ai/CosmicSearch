#!/usr/bin/env bash
# Zero-downtime deploy.
#
# `apps-platform app deploy` sends 100% of traffic to the new revision the
# moment it is created. That revision opens its port in seconds but has no
# corpus for ~15 minutes, so every deploy used to mean ~15 minutes of 503s.
#
# Cloud Run cannot be told to wait: a startup probe is capped at 240s
# (failureThreshold * periodSeconds), and the corpus screen alone takes 239s
# before the ~8 minute model load. So readiness cannot gate traffic, and the
# switch has to be made from outside.
#
#   1. pin traffic to the revision serving now
#   2. deploy -- the new revision is created with 0% of traffic
#   3. give it a traffic tag, which is what makes Cloud Run start its
#      min-instances at all: a revision outside the traffic split is not
#      started for min-instances alone, so without this it would never warm
#   4. wait for it to log its corpus ready
#   5. shift traffic to it, instantly, already warm
#
# The old revision serves throughout. If step 3 never succeeds, traffic never
# moves and the deploy is abandoned with the service still up.
#
# Usage:  ./deploy.sh                       # cars   (project.toml)
#         ./deploy.sh project-trucking.toml # trucks (project-trucking.toml)
set -euo pipefail

CONFIG="${1:-project.toml}"
REGION="us-west1"
GCP_PROJECT="experimental-apps-v2"
SERVICE="$(sed -n 's/^name *= *"\(.*\)"/\1/p' "$CONFIG" | head -1)"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"

[ -n "$SERVICE" ] || { echo "could not read service name from $CONFIG" >&2; exit 1; }
run() { gcloud run services "$@" --region "$REGION" --project "$GCP_PROJECT"; }

CURRENT="$(run describe "$SERVICE" --format 'value(status.latestReadyRevisionName)')"
echo "service        : $SERVICE"
echo "serving now    : $CURRENT"

# 1. Pin. Until this is undone, any revision created gets no traffic.
echo "==> pinning traffic to $CURRENT"
run update-traffic "$SERVICE" --to-revisions "$CURRENT=100" >/dev/null

# Whatever happens next, traffic stays where it is unless we move it
# deliberately -- a failed deploy must not leave the service on a cold revision.
trap 'echo "!! deploy did not complete; traffic still on $CURRENT"' ERR

# 2. Build and create the new revision. It starts with 0% of traffic.
echo "==> deploying (new revision will take no traffic)"
apps-platform app deploy ${1:+-c "$CONFIG"}

NEW="$(run describe "$SERVICE" --format 'value(status.latestCreatedRevisionName)')"
if [ "$NEW" = "$CURRENT" ]; then
  echo "no new revision was created; nothing to do"
  exit 0
fi
echo "new revision   : $NEW"

# 3. Tag it. Revision-level min-instances only start when the revision is
#    referenced in the traffic split or carries a tag -- and step 1 removed it
#    from the split. The tag is what gets the corpus loading.
echo "==> tagging $NEW as 'candidate' so it starts warming"
run update-traffic "$SERVICE" --set-tags "candidate=$NEW" >/dev/null

# 4. Wait for the corpus. This is the whole point: the switch happens only
#    once the new revision can actually answer a search.
echo "==> waiting for $NEW to load its corpus (up to ${READY_TIMEOUT_S}s)"
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
while :; do
  log="$(gcloud logging read \
      "resource.labels.revision_name=\"$NEW\" AND (textPayload:\"corpus ready\" OR textPayload:\"corpus load failed\")" \
      --project "$GCP_PROJECT" --limit 1 --freshness=1h --format 'value(textPayload)' 2>/dev/null || true)"
  case "$log" in
    *"corpus ready"*)
      echo "    $log"; break ;;
    *"corpus load failed"*)
      echo "!! $log" >&2
      echo "!! traffic left on $CURRENT" >&2
      run update-traffic "$SERVICE" --clear-tags >/dev/null || true
      exit 1 ;;
  esac
  [ "$(date +%s)" -lt "$deadline" ] || {
    echo "!! $NEW did not report a corpus within ${READY_TIMEOUT_S}s; traffic left on $CURRENT" >&2
    exit 1; }
  sleep 20
done

# 5. Switch. The revision is warm, so this is instant.
echo "==> shifting traffic to $NEW"
run update-traffic "$SERVICE" --to-revisions "$NEW=100" >/dev/null
# Drop the tag: it has done its job, and a tagged revision is kept running
# (and billed) forever even with no traffic.
run update-traffic "$SERVICE" --clear-tags >/dev/null
trap - ERR
echo "done: $SERVICE now serving $NEW"
