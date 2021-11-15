import urllib.request


if __name__ == "__main__":
    data = urllib.request.urlopen("https://www.zhihu.com/question/46956394")
    print(data.read())
