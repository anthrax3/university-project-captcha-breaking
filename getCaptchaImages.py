import urllib.request

url = "http://webspamprotect.com/PHPCaptcha/wspcaptcha.php"

for i in range(1000,2000):
	request = urllib.request.Request(url)
	response = urllib.request.urlopen(request)
	f = response.read();
	open(str(i)+".png","wb").write(f);


'''
import urllib.request
import lxml.html

urllib.urlretrieve('http://webspamprotect.com/PHPCaptcha/wspcaptcha.php','captcha.bmp')

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0.1) Gecko/2010010' \
    '1 Firefox/4.0.1',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language':'en-us,en;q=0.5',
    'Accept-Charset':'ISO-8859-1,utf-8;q=0.7,*;q=0.7'}

req = urllib.request('http://webspamprotect.com/PHPCaptcha/wspcaptcha.php', None,
                      headers)
f = urllib.urlopen(req)
page = f.read()

print(page)

tree = lxml.html.fromstring(page)
imgurl = "http://www.amaderforum.com/" + \
      tree.xpath(".//img[@id='imagereg']")[0].get('src')

req = urllib.Request(imgurl, None, headers)
f = urllib.urlopen(req)
img = f.read()

open('out.jpg', 'wb').write(img)
'''
