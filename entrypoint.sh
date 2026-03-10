#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  Container Entrypoint — UID/GID Alignment
#═══════════════════════════════════════════════════════════════════════════════
#  Dynamically adjusts the container user's UID/GID to match the owner of
#  the bind-mounted directories.  This lets *any* host user read/write
#  bind-mounts without file-permission errors and without rebuilding.
#
#  When HOST_UID / HOST_GID are passed (e.g., via `docker run -e`), those
#  values are used directly.  Otherwise the script probes the owner of
#  /home/workspace/sharc/examples (a representative bind-mount).
#═══════════════════════════════════════════════════════════════════════════════

set -e

TARGET_USER="${CONTAINER_USER:-admin}"
PROBE_DIR="/home/workspace/sharc/examples"

# ── Determine desired UID/GID ────────────────────────────────────────────────
if [ -z "$HOST_UID" ] || [ -z "$HOST_GID" ]; then
    if [ -d "$PROBE_DIR" ]; then
        HOST_UID="$(stat -c '%u' "$PROBE_DIR")"
        HOST_GID="$(stat -c '%g' "$PROBE_DIR")"
    fi
fi

# Fall back to image defaults if nothing was detected (no bind mount)
HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"

# ── Get current UID/GID of the target user ───────────────────────────────────
CUR_UID="$(id -u "$TARGET_USER" 2>/dev/null || echo "")"
CUR_GID="$(id -g "$TARGET_USER" 2>/dev/null || echo "")"

# ── Align GID ────────────────────────────────────────────────────────────────
if [ -n "$CUR_GID" ] && [ "$CUR_GID" != "$HOST_GID" ]; then
    # If another group already uses the target GID, rename it to avoid conflict
    EXISTING_GROUP="$(getent group "$HOST_GID" | cut -d: -f1 || true)"
    if [ -n "$EXISTING_GROUP" ] && [ "$EXISTING_GROUP" != "$(id -gn "$TARGET_USER")" ]; then
        groupmod -g 99999 "$EXISTING_GROUP" 2>/dev/null || true
    fi
    groupmod -g "$HOST_GID" "$(id -gn "$TARGET_USER")" 2>/dev/null || true
fi

# ── Align UID ────────────────────────────────────────────────────────────────
if [ -n "$CUR_UID" ] && [ "$CUR_UID" != "$HOST_UID" ]; then
    # If another user already uses the target UID, shift it out of the way
    EXISTING_USER="$(getent passwd "$HOST_UID" | cut -d: -f1 || true)"
    if [ -n "$EXISTING_USER" ] && [ "$EXISTING_USER" != "$TARGET_USER" ]; then
        usermod -u 99999 "$EXISTING_USER" 2>/dev/null || true
    fi
    usermod -u "$HOST_UID" "$TARGET_USER" 2>/dev/null || true
fi

# ── Fix ownership of the user's home directory ───────────────────────────────
chown -R "$HOST_UID:$HOST_GID" "/home/$TARGET_USER" 2>/dev/null || true

# ── Drop from root → TARGET_USER and exec the requested command ──────────────
exec gosu "$TARGET_USER" "$@"
