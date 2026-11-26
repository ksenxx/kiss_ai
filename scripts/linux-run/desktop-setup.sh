#!/bin/sh
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Make this Linux machine the machine of a new desktop user, for the checks
# of the Linux installer: scripts/check-linux-run.sh runs it in a container
# and scripts/check-linux-vm.sh in a virtual machine, as root.
#
#   sh desktop-setup.sh [display]
#
# The image of a container or of a cloud server lacks what a desktop
# install of the distribution has: perl, lsof, the libraries Chromium
# needs and a download tool.  On Debian and Ubuntu that tool is wget
# alone: their desktops ship no curl and no git, so both are removed where
# the image has them.  With "display" the machine also gets what the check
# of the app's window needs: a display to open it on (Xvfb), a file
# manager for the launcher's folder and the tools that look at the result.
#
# Last it adds the user "tester", who has no sudo, with a Downloads
# folder, and prints what the machine runs.

set -eu

display="${1:-}"
. /etc/os-release
case "$ID" in
    ubuntu|debian)
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends wget ca-certificates perl lsof procps \
            libasound2t64 libatk-bridge2.0-0t64 libatk1.0-0t64 libatspi2.0-0t64 libcairo2 \
            libcups2t64 libdbus-1-3 libdrm2 libgbm1 libglib2.0-0t64 libnspr4 libnss3 \
            libpango-1.0-0 libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 libxfixes3 \
            libxkbcommon0 libxrandr2 > /dev/null
        [ -z "$display" ] || apt-get install -y -qq --no-install-recommends \
            xvfb x11-utils imagemagick desktop-file-utils xdg-utils pcmanfm > /dev/null
        apt-get remove -y -qq curl git > /dev/null
        ;;
    fedora)
        dnf -y -q install wget2-wget perl-interpreter lsof procps-ng \
            alsa-lib atk at-spi2-atk cairo cups-libs dbus-libs libdrm mesa-libgbm glib2 nspr \
            nss pango libX11 libxcb libXcomposite libXdamage libXext libXfixes libxkbcommon \
            libXrandr > /dev/null
        [ -z "$display" ] || dnf -y -q install \
            xorg-x11-server-Xvfb xwininfo ImageMagick desktop-file-utils xdg-utils pcmanfm > /dev/null
        ;;
    opensuse*)
        zypper -n -q install curl wget perl lsof procps findutils gawk tar gzip shadow \
            libasound2 libatk-1_0-0 libatk-bridge-2_0-0 libatspi0 libcairo2 libcups2 \
            libdbus-1-3 libdrm2 libgbm1 libglib-2_0-0 mozilla-nspr mozilla-nss \
            libpango-1_0-0 libX11-6 libxcb1 libXcomposite1 libXdamage1 libXext6 libXfixes3 \
            libxkbcommon0 libXrandr2 > /dev/null
        [ -z "$display" ] || zypper -n -q install \
            xorg-x11-server-Xvfb xwininfo ImageMagick desktop-file-utils xdg-utils pcmanfm > /dev/null
        ;;
    arch)
        pacman -Syu --noconfirm --needed wget perl lsof procps-ng \
            alsa-lib at-spi2-core cairo libcups dbus libdrm mesa glib2 nspr nss pango libx11 \
            libxcb libxcomposite libxdamage libxext libxfixes libxkbcommon libxrandr > /dev/null
        [ -z "$display" ] || pacman -S --noconfirm --needed \
            xorg-server-xvfb xorg-xwininfo imagemagick desktop-file-utils xdg-utils pcmanfm > /dev/null
        ;;
    *)
        echo "no package list for the distribution '$ID'" >&2
        exit 1
        ;;
esac
useradd -m -s /bin/bash tester
mkdir /home/tester/Downloads
chown tester: /home/tester/Downloads
echo "The machine runs $PRETTY_NAME ($(uname -m))"
