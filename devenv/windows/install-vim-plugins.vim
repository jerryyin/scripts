set nocompatible
set eventignore=all
execute 'set runtimepath^=' . fnameescape($DEVENV_VIM_HOME . '/.vim')
execute 'source ' . fnameescape($DEVENV_RC_ROOT . '/vim/.vim/config/plugins.vim')
if $DEVENV_UPDATE ==# '1'
  PlugUpdate --sync
elseif !empty(filter(copy(values(g:plugs)), '!isdirectory(v:val.dir)'))
  PlugInstall --sync
else
  echom 'SKIP Vim plugins (installed)'
endif
if !empty(filter(values(g:plugs), '!filereadable(v:val.dir . ''/.git/HEAD'')'))
  cquit 1
endif
let s:choices = $DEVENV_VIM_HOME . '/.vim/coc-extensions.json'
let s:extensions = json_encode(g:coc_global_extensions)
if !filereadable(s:choices) || join(readfile(s:choices), '') !=# s:extensions
  call writefile([s:extensions], s:choices)
endif
cquit 0
