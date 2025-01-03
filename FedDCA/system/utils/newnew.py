import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter
import string

# 加载spaCy的英语模型
nlp = spacy.load('en_core_web_sm')


# 1. 获取页面内容
def get_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    return response.text


# 2. 解析网页，获取文章链接
def get_article_links(base_url):
    page_content = get_page(base_url)
    soup = BeautifulSoup(page_content, 'html.parser')

    # 提取所有新闻链接
    article_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/culture/art') and href not in article_links:
            article_links.append(base_url + href)

    return article_links


# 3. 获取单篇文章内容
def get_article_content(url):
    page_content = get_page(url)
    soup = BeautifulSoup(page_content, 'html.parser')

    # 找到文章正文部分 (通常是 <div class="article-body"> 或类似标签)
    paragraphs = soup.find_all('div', class_='article-body')
    article_text = ''

    for paragraph in paragraphs:
        article_text += paragraph.get_text()

    return article_text


# 4. 处理文本，提取英文单词
def clean_and_tokenize(text):
    # 转为小写，去掉标点符号
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 使用spaCy进行分词
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]

    return words


# 5. 计算高频词
def get_frequent_words(base_url):
    article_links = get_article_links(base_url)
    all_words = []

    for link in article_links:
        print(f"Processing {link}")
        article_content = get_article_content(link)
        words = clean_and_tokenize(article_content)
        all_words.extend(words)

    # 计算词频
    word_counts = Counter(all_words)
    return word_counts


# 主程序
base_url = "https://www.chinadaily.com.cn/culture"  # 主页链接
word_counts = get_frequent_words(base_url)

# 输出高频词
print("Top 10 Frequent Words:")
for word, count in word_counts.most_common(10):
    print(f"{word}: {count}")
