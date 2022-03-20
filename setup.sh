mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor = '#6998AB'\n\
backgroundColor = '#1A374D'\n\
secondaryBackgroundColor = '#406882'\n\
textColor = '#FFF'\n\
font = 'sans serif'\n\
" > ~/.streamlit/config.toml