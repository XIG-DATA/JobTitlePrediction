#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests, time
from bs4 import BeautifulSoup
import codecs
from baidubaike import Page
from baidubaike.exception import PageError
from pullword import pullword
my_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.125 Safari/537.36'}

#User-Agent:Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53
import sys  
reload(sys)  
sys.setdefaultencoding('utf-8')
#items = [u'结构工程师',u'营业担当兼翻译']

def craw_baike(url):
	session = requests.session()
	req = session.get(url, headers=my_headers)
	req.encoding = 'utf-8'
	soup = BeautifulSoup(req.text, "html.parser")
	item = soup.find('div', {'class': 'para'})
	try:
		txt = item.get_text().encode('utf-8')
	except AttributeError:
		txt = 'none'
	return txt


def get_baike_pages(items):
	item_list = []
	for i, item in enumerate(items):
		time.sleep(5)
		if i% 10 == 0:
			print i, item
		try:
			item = item.strip()
			item_page = Page(item)
			dicts = item_page.get_info()
			url = dicts['url']
			item_list.append([ item, craw_baike(url)])
		except PageError:
			word_list = pullword(item, threshold=0.7)
			newitem =  word_list[0][0]
			try:
				new_item_page = Page(newitem)
				newdicts = new_item_page.get_info()
				url = newdicts['url']
				item_list.append([ item, craw_baike(url)])
			except PageError:
				print 'sorry, I do not understand'
	
	fp = codecs.open('major_decription.txt', 'wb', 'utf8') 
	for item, des in item_list:
		try :
			fp.write(item +  "::" + des + '\n')
		except UnicodeDecodeError:
			print item, des
	return item_list

def get_major_description(fn):
	lines = open(fn).readlines()
	results = get_baike_pages(lines)

get_major_description('../major.txt')