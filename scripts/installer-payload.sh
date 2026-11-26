# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# The payload of a KISS Sorcar installer: sourced by scripts/build-macos-pkg.sh
# (the Mac package), scripts/build-windows-exe.sh (the Windows setup
# program) and scripts/build-linux-run.sh (the Linux installer), which all
# carry the same copy of the sources and run its install.sh on the user's
# machine.
#
# The script that sources this file sets PROJECT_DIR (the checkout) and
# defines ``die``.

project_version() {
    # Print the version of this checkout (kiss/core/_version.py).
    sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$PROJECT_DIR/src/kiss/core/_version.py"
}

# list_app_files and project_is_git_checkout of install.sh: the payload
# holds exactly the files install.sh installs, by its own list of them.
eval "$(sed -n '/^# BEGIN: kiss-app-files/,/^# END: kiss-app-files/p' "$PROJECT_DIR/install.sh")"

list_payload_files() {
    # Print, relative to the checkout, every file of the payload: the
    # app's files, and install.sh with the scripts it installs (rsorcar,
    # sorcar-docker with its Dockerfile) or runs (scripts/).
    local path
    list_app_files
    for path in install.sh rsorcar sorcar-docker Dockerfile; do
        if [ -f "$PROJECT_DIR/$path" ]; then
            echo "$path"
        fi
    done
    (
        cd "$PROJECT_DIR"
        if project_is_git_checkout; then
            git ls-files scripts
        else
            find scripts -type f ! -name .DS_Store
        fi
    )
}

stage_payload() {
    # Copy the payload's files into directory $1.
    local root="$1" list="$1.files"
    mkdir -p "$root"
    list_payload_files > "$list"
    # COPYFILE_DISABLE: no "._name" files for extended attributes.
    (cd "$PROJECT_DIR" && COPYFILE_DISABLE=1 tar cf - -T "$list") | (cd "$root" && tar xf -)
    rm -f "$list"
    [ -f "$root/src/kiss/__init__.py" ] || die "no files of the app found under $PROJECT_DIR/src/kiss"
    # The commit the Settings panel shows next to the version.  install.sh
    # writes it from git for a checkout; the payload has no git history
    # (and the user's machine may have no git), so it ships the file.
    if project_is_git_checkout; then
        git -C "$PROJECT_DIR" rev-parse --short HEAD > "$root/src/kiss/core/_commit.txt"
    fi
    # The same modes whatever the umask of the machine that builds.
    chmod -R u+rwX,go+rX,go-w "$root"
}
