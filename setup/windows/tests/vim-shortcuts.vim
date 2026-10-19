let g:auto_save = 0
set clipboard=
let s:checks = []
let s:errors = []
let s:project = getcwd()
function! s:Check(ok, name) abort
  call add(s:checks, {'passed': !!a:ok, 'name': a:name})
endfunction
function! s:Keys(keys) abort
  call feedkeys(a:keys, 'xt')
endfunction
function! s:Search(keys, label) abort
  cclose
  execute 'edit ' . fnameescape(s:project . '/first.txt')
  call cursor(1, 1)
  call setqflist([])
  call s:Keys(a:keys)
  sleep 800m
  let matches = filter(getqflist(), 'v:val.valid')
  call s:Check(len(matches) == 2, a:label . ' finds two matches, excludes build and word suffix')
  call s:Check(sort(map(copy(matches), 'fnamemodify(bufname(v:val.bufnr), ":t")')) == ['first.txt', 'second file.txt'], a:label . ' handles file names with spaces')
endfunction
try
  let slash = &shellslash
  call s:Search('\sa', '\sa searches the word under the cursor')
  call s:Check(&shellslash == slash, 'search preserves the shell configuration')
  call timer_start(200, {-> feedkeys("\"value with spaces\"\<CR>", 't')})
  call s:Search('\ss', '\ss searches a quoted phrase')
catch
  call add(s:errors, v:exception . ' at ' . v:throwpoint)
endtry
try
  cclose
  execute 'edit ' . fnameescape($VIM_AUDIT_SCRIPTS . '/setup/windows/bootstrap.ps1')
  call search('Ensure-Wsl')
  call setqflist([])
  call s:Keys('\sg')
  sleep 1000m
  call s:Check(!empty(filter(getqflist(), 'v:val.valid && v:val.text =~# "Ensure-Wsl"')), '\sg searches tracked Git files')
  call s:Keys('\qo')
  call s:Check(&buftype ==# 'quickfix', '\qo opens search results')
  call s:Keys('\qc')
  call s:Check(empty(filter(getwininfo(), 'v:val.quickfix')), '\qc closes search results')
  execute 'cd ' . fnameescape(s:project)
  execute 'edit ' . fnameescape(s:project . '/first.txt')
  let first = win_getid()
  vsplit
  call s:Keys("\<C-h>")
  call s:Check(win_getid() == first, 'Ctrl-h moves to left split')
  call s:Keys("\<C-l>")
  call s:Check(win_getid() != first, 'Ctrl-l moves to right split')
  split
  let bottom = win_getid()
  call s:Keys("\<C-k>")
  call s:Check(win_getid() != bottom, 'Ctrl-k moves to upper split')
  call s:Keys("\<C-j>")
  call s:Check(win_getid() == bottom, 'Ctrl-j moves to lower split')
  only
  call s:Keys('\fn')
  call s:Check(!empty(filter(getwininfo(), 'getbufvar(v:val.bufnr, "&filetype") ==# "nerdtree"')), '\fn opens the file tree')
  call s:Keys('\fn')
  call s:Check(empty(filter(getwininfo(), 'getbufvar(v:val.bufnr, "&filetype") ==# "nerdtree"')), '\fn closes the file tree')
  let g:startify_session_dir = $VIM_AUDIT_ROOT . '/sessions'
  call mkdir(g:startify_session_dir, 'p')
  call s:Keys('\sv')
  call s:Check(filereadable(g:startify_session_dir . '/' . GetDefaultSession()), '\sv saves a session')
  let g:startify_session_savevars = []
  call s:Keys('\sl')
  call s:Check(expand('%:t') ==# 'first.txt', '\sl reloads the session')
  call s:Keys('\sd')
  call s:Check(!filereadable(g:startify_session_dir . '/' . GetDefaultSession()), '\sd deletes the saved session')
catch
  call add(s:errors, v:exception . ' at ' . v:throwpoint)
endtry
try
  cclose
  execute 'cd ' . fnameescape(s:project)
  let g:Lf_UseCache = 0
  runtime autoload/leaderf.vim
  py3 << EOF
import traceback
try:
    from leaderf.fileExpl import FileExplorer
    vim.vars['audit_files'] = list(FileExplorer().getContent(arguments={}, refresh=True))
except Exception:
    vim.vars['audit_files'] = []
    vim.vars['audit_python_error'] = traceback.format_exc()
EOF
  if exists('g:audit_python_error') | call add(s:errors, g:audit_python_error) | endif
  call s:Check(index(map(copy(g:audit_files), 'fnamemodify(v:val, ":t")'), 'second file.txt') >= 0, 'LeaderF file discovery returns a file with spaces')
  execute 'cd ' . fnameescape($VIM_AUDIT_RC)
  py3 vim.vars['audit_git_files'] = list(FileExplorer().getContent(arguments={}, refresh=True))
  call s:Check(!empty(filter(copy(g:audit_git_files), 'substitute(v:val, "\\\\", "/", "g") =~# "vim/\\.vimrc$"')), 'LeaderF lists tracked files in the real rc_files repository')
  execute 'cd ' . fnameescape(s:project)
catch
  call add(s:errors, v:exception . ' at ' . v:throwpoint)
endtry
call writefile([json_encode({'checks': s:checks, 'errors': s:errors, 'files': get(g:, 'audit_git_files', []), 'fileCommand': get(g:, 'Lf_Debug_Cmd', ''), 'messages': execute('messages')})], $VIM_AUDIT_RESULT)
function! s:ReportUi(timer) abort
  let visible = filter(popup_list(), 'get(popup_getpos(v:val), "visible", 0)')
  call writefile([json_encode({'visible': len(visible), 'file': expand('%:p'), 'line': line('.'), 'cocReady': get(g:, 'coc_service_initialized', 0), 'diagnosticErrors': get(get(b:, 'coc_diagnostic_info', {}), 'error', 0), 'signify': get(get(b:, 'sy', {}), 'updated_by', ''), 'bridgePresent': join(getline(1, '$')) =~# 'paste_bridge_marker'})], $VIM_AUDIT_ROOT . '/ui.json')
endfunction
autocmd VimEnter * call timer_start(100, function('s:ReportUi'), {'repeat': -1})
