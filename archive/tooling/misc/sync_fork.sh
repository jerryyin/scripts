fork_url=https://github.com/deven-amd/dlaicourse
upstream_url=https://github.com/lmoroney/dlaicourse

mkdir temp && \
    cd temp && \
    git clone $fork_url repo && \
    cd repo && \
    git remote add upstream $upstream_url && \
    git fetch upstream && \
    git merge upstream/master && \
    git push origin master
