shopt -s extglob  # enable
cat !(readme.md|contrib.rst|licence.txt)
shopt -u extglob  # disable

