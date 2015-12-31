# coding: utf-8
# 谷歌翻译Python接口
# http://www.site-digger.com
# hello@site-digger.com

import sys
import re
import gzip
import socket
import urllib
import urllib2
from StringIO import StringIO
try:
    import simplejson as json
except ImportError:
    import json

# set default timeout
socket.setdefaulttimeout(30)

class Download:
    """HTTP交互类
    """
    def __init__(self):
        self.user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:5.0) Gecko/20100101 Firefox/5.0'
        self.opener = None

    def fetch(self, url, headers=None, data=None, proxy=None, user_agent=None):
        """下载指定URL，返回内容
        支持GET和POST请求
        支持GZIP压缩
        """
        # create opener with headers
        opener = self.opener or urllib2.build_opener()
        if proxy:
            if url.lower().startswith('https://'):
                opener.add_handler(urllib2.ProxyHandler({'https': proxy}))
            else:
                opener.add_handler(urllib2.ProxyHandler({'http': proxy}))
        default_headers =  {'User-Agent': self.user_agent , 'Accept-encoding': 'gzip, deflate', 'Referer': url}
        headers = headers and default_headers.update(headers) or default_headers
        
        if isinstance(data, dict):
            data = urllib.urlencode(data) 
        try:
            response = opener.open(urllib2.Request(url, data, headers))
            content = response.read()
            if response.headers.get('content-encoding') == 'gzip':
                # data came back gzip-compressed so decompress it          
                content = gzip.GzipFile(fileobj=StringIO(content)).read()
        except Exception, e:
            # so many kinds of errors are possible here so just catch them all
            sys.stderr.write('Download exception: "%s"\n' % str(e))
            content = ''
        return content

def clean_google_json(googlejson):
    """清理谷歌GWT返回数据
    """
    # pass through result and turn empty elements into nulls
    instring = False
    inescape = False
    lastchar = ''
    output = ""
    for char in googlejson:
        # toss unnecessary whitespace
        if ((not instring) and (re.match(r'\s+', char))):
            continue

        # handle strings
        if instring:
            if inescape:
                output += char
                inescape = False
            elif char == '\\':
                output += char
                inescape = True
            elif char == '"':
                output += char
                instring = False
            else:
                output += char
            lastchar = char
            continue

        if char == '"':
            output += char
            instring = True
        elif char == ',':
            if lastchar == ',' or lastchar == '[' or lastchar == '{':
                output += 'null'
            output += char
        elif char == ']' or char == '}':
            if lastchar == ',':
                output += 'null'
            output += char
        else:
            output += char
        lastchar = char

    return output.replace('\n', '\\n')

def translate(sl, tl, content, proxy=None):
    """提交翻译请求，返回解析结果
    sl - 源语言
    tl - 目标语言
    content - 要翻译的内容，必须为UTF-8编码
    proxy - 要使用的代理
    返回翻译的结果
    """
    D = Download()
    result = ''
    if content:
        url = 'http://translate.google.cn/translate_a/t'
        post_data = {}
        post_data['q'] = content
        post_data['rom'] = '1'
        post_data['oe'] = 'UTF-8'
        post_data['multires'] = '1'
        post_data['oc'] = '1'
        post_data['otf'] = '2'
        post_data['tsel'] = '4'
        post_data['tl'] = tl
        post_data['client'] = 't'
        post_data['hl'] = 'zh-CN'
        post_data['sl'] = sl
        post_data['sc'] = '1'
        post_data['ssel'] = '4'
        post_data['ie'] = 'UTF-8'
        google_confusing_json = D.fetch(url=url, data=post_data, proxy=proxy)
        google_json_string = clean_google_json(google_confusing_json)
        try:
            google_json_data = json.loads(google_json_string)
        except Exception, e:
            sys.stderr.write('Failed to parse json string: "%s"\n' % str(e))
        else:
            lines = []
            for item in google_json_data[0]:
                lines.append(item[0])
            result = ''.join(lines)
    return result

if __name__ == '__main__':
    print translate(sl='zh-CN', tl='en', content='毛主席万岁！\n中国人民大团结万岁！')