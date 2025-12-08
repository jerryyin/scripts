HG_CONFIG_FILE=$HOME/.hgrc

echo "# hg configuration file" > $HG_CONFIG_FILE
echo "" >> $HG_CONFIG_FILE

echo "[ui]" >> $HG_CONFIG_FILE
echo "username = Deven Desai <deven.desai.amd@gmail.com>" >> $HG_CONFIG_FILE
echo "" >> $HG_CONFIG_FILE

echo "[extensions]" >> $HG_CONFIG_FILE
echo "color =" >> $HG_CONFIG_FILE
echo "histedit =" >> $HG_CONFIG_FILE
echo "rebase =" >> $HG_CONFIG_FILE
echo "strip =" >> $HG_CONFIG_FILE
echo "" >> $HG_CONFIG_FILE

# echo "[]" >> $HG_CONFIG_FILE
# echo "" >> $HG_CONFIG_FILE
# echo "" >> $HG_CONFIG_FILE
# echo "" >> $HG_CONFIG_FILE
# echo "" >> $HG_CONFIG_FILE
# echo "" >> $HG_CONFIG_FILE

# https://www.mercurial-scm.org/wiki/HisteditExtension
