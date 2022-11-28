from os import path
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import re
import emoji
from soynlp.normalizer import repeat_normalize
from kiwipiepy import Kiwi
import json
import logging


def save_text(save_path, text_list, configs=None):
    check_exists = "a" if path.exists(save_path) else "w"
    with open(save_path, check_exists) as f:
        if configs:  # in case text_list is Sentence dataset
            for sent in text_list:
                txt = preprocess_data(sent.text)
                if txt:
                    f.write(f'{configs["keyword"]}\t{configs["portal"]}\t{configs["date"]}\t{txt}\n')
        else:  # in case text_list is Link dataset
            for link in text_list:
                f.write(link + '\n')
        f.close()
    logging.info(f"{save_path.split('/')[-1]} has successfully saved!")


def preprocess_data(data):
    def clean(x, patterns):
        for pattern in patterns:
            x = pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        if x:
            return x
        return None

    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern_list = [
        re.compile(f'[^ .,?!/@$%~％·∼()x00-\x7Fㄱ-힣{emojis}]+'),
        re.compile(f'[0-9]+.'),
        re.compile(f'[0-9]+:[0-9]+'),
        re.compile(f'[a-zA-Z]+:?.?[a-zA-Z]+.?[a-zA-Z]?'),
        re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        ),
    ]

    return clean(data, pattern_list)


def collect_link(portals, keywords, dates, scroll_configs):
    def get_naver_links(keywords, dates, max_scroll):
        date_from, date_to = dates["from"], dates["to"]
        etc_date_str = f"from{date_from.strip().replace('.', '')}to{date_from.strip().replace('.', '')}"
        base_url = "https://search.naver.com/search.naver"

        for keyword in keywords:
            save_path = f"네이버_{keyword}_link.txt"
            # 네이버 카페(VIEW) / 검색 키워드 / 기간 설정 / 관련도 높은 순 검색
            url = f"{base_url}?where=article&query={keyword}&ie=utf8&st=rel&date_option=99&date_from={date_from}&date_to={date_to}&board=&srchby=text&dup_remove=1&cafe_url=&without_cafe_url=&sm=tab_opt&nso=so%3Ar%2Cp%3A{etc_date_str}&nso_open=1&t=0&mson=0&prdtype=0"
            driver = webdriver.Edge(executable_path=r"./msedgedriver")
            time.sleep(3)
            driver.get(url)

            count = 1
            last_height = driver.execute_script("return document.body.scrollHeight")
            while count < max_scroll:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                count += 1

            index = 1
            link_list = []
            while True:
                try:
                    link = driver.find_element(
                        By.XPATH,
                        f'//*[@id="_view_review_body_html"]/div/more-contents/div/ul/li[{str(index)}]/div[1]/div/a'
                    ).get_attribute("href")
                    link_list.append(link)
                    index += 1
                except:
                    break

            print(keyword + ': ' + str(len(link_list)))
            save_text(save_path, link_list)
            driver.quit()

    for portal in portals:
        if portal == "네이버":
            logging.info(f"{portal} link collection started...")
            max_scroll = scroll_configs[portal]
            get_naver_links(keywords, dates, max_scroll)


def collect_sentences(keywords, portals):
    def get_naver_sentences(portal, keywords):
        for keyword in keywords:
            driver = webdriver.Edge(executable_path=r"./msedgedriver")
            driver.maximize_window()
            time.sleep(2)

            save_path = f'{portal}_{keyword}_sentences.txt'
            link_file = open(f'./{portal}_{keyword}_link.txt', 'r')
            link_list = link_file.readlines()
            for link in link_list:
                try:
                    driver.get(link)
                    time.sleep(2.5)
                    driver.switch_to.frame("cafe_main")

                    source = bs(driver.page_source, 'html.parser')
                    date = source.select("div.article_info > span.date")[0].text.split(' ')[0][:-1]
                    title_source = source.select("h3.title_text")
                    main_source = source.select("div.article_container > div.article_viewer")
                    comments_source = source.select("div.comment_box > div.comment_text_box")
                    source_list = [*title_source, *main_source, *comments_source]

                    for i in source_list:
                        txt = i.text.strip()
                        sentences = kiwi.split_into_sents(txt, normalize_coda=True)
                        configs = {
                            "keyword": keyword,
                            "portal": portal,
                            "date": date
                        }
                        save_text(save_path, sentences, configs)
                except:
                    logging.exception(f"{link} didn't saved...")
                    continue
            driver.close()
            driver.quit()

    kiwi = Kiwi()  # 문장 분리기
    for portal in portals:
        if portal == "네이버":
            logging.info(f"{portal} sentence collection started...")
            get_naver_sentences(portal, keywords)


def crawler():
    with open('./config.json', 'r') as config_file:
        configs = json.load(config_file)

    logging.basicConfig(
        filename=configs['logger'],
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    keywords = configs['keywords']
    portals = configs['portals']
    dates = configs['dates']
    scroll_configs = configs['max_scroll']

    collect_link(keywords, portals, dates, scroll_configs)
    time.sleep(60)
    print("Text Scrape process will be start in 60 seconds...")
    collect_sentences(keywords, portals)

            
if __name__ == "__main__":
    crawler()
