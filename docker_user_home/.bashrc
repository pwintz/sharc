#!/bin/bash
source /opt/ros/foxy/setup.bash

# Modify the bash prompt as follows:
# * Show the name of the machine in green if last command succeeded, otherwise red.
# * Print the working directory relative to HOME
# * Line break so that command can use full width of window.
# PS1='\[\033]0;Love the unloved:$PWD\007\]' # set window title
PS1="\[\`if [[ \$? = "0" ]]; then echo \e[32m\h\e[0m; else echo \e[31m\h\e[0m ; fi\`"
PS1="$PS1:\w"                             # ":<current directory>"
PS1="$PS1"'\n\$ '                         # new line and "$"

alias ..="cd .."
alias ...="cd ../.."
alias ....="cd ../../.."
alias .....="cd ../../../.."
alias ......="cd ../../../../.."

alias l="ls -lh"
alias ll="ls -lah"

function mkcd() {
	mkdir $1 
	cd $1
}


# GIT
# if git is installed...
# if hash git 2>/dev/null; then
if command -v git &> /dev/null
then

    alias tig="tig --all"

    alias f='git fetch --all';
    alias d='git diff';
    alias ds='git diff --staged';
    alias a='git add';
    alias ap='git add -p';
    alias rp='git reset -p';
    alias st-keepindex='git stash --keep-index';
    alias st='git stash';
    alias cm='git commit';
    alias cmm='git commit -m';
    alias cma='git commit --amend';
    alias co='git checkout';
    alias cop='git checkout -p';
    alias cob='git checkout -b';
    alias mergeff='git merge --ff'
    alias push='git push'
    alias t="tig"
   
    unset s
    function s() {
        # If in a git repo...
        if git rev-parse 2> /dev/null; then
            ls
            git status --show-stash
            (git fetch --all | tail +2 &) # Update the repo asynchronously. Only prints if new objects were found. 
        else
            echo "The working directory $(pwd) is not a git repository. It contains: "
            ls
        fi
    }
    
    alias pushu="git push --set-upstream origin $(git branch 2>/dev/null | awk '/^\* / { print $2 }')"
else
	echo "Git not  found."
    alias s="ls"
fi