In case AutoDL has errors, use the following codes:
```
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# if issue persists
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
```
