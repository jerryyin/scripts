git config --global --unset user.name
git config --global --unset user.email

cd ~ && rm -rf \
  .ssh \
  .git \
  .gitignore \
  .gitconfig \
  .tmux.conf \
  .emacs \
  .vimrc \
  .vim \
  .zshrc \
  .oh-my-zsh \
  .zsh_history \
  .zsh \
  .cache
  
mv .bashrc.bkp .bashrc
