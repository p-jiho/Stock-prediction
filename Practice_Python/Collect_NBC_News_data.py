# Collect NBC News Data
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

year = list(range(2012, 2023))
month = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
         "december"]
month_number = list(range(1, 13))
month_number = ["0" + str(month_number[i]) if len(str(month_number[i])) == 1 else str(month_number[i]) for i in
                range(len(month_number))]

for i in year:
    for j in range(12):
        main_page = requests.get("https://www.nbcnews.com/archive/articles/" + str(i) + "/" + month[j])
        soup = bs(main_page.text, "html.parser")
        elements = soup.select('body > div.Layout > main > a')
        href = [element.attrs["href"] for element in elements]
        title = [element.text for element in elements]

        f = open("../../data/NBC_News/" + str(i) + "_" + month_number[j] +".txt", 'w', encoding='UTF-8')

        for k in range(len(href)):
            page = requests.get(href[k])
            soup = bs(page.text, "html.parser")
            catagory = soup.select(
                "#content > div:nth-child(7) > div > article > section > div.article-hero__bg-container > header > aside > div > a > span")
            catagory = [catagory[0].text if catagory != [] else ""]
            date = soup.select(
                "#content > div:nth-child(7) > div > article > div > div > div.article-body__section.layout-grid-container.article-body__last-section.article-body__first-section > div.article-body.layout-grid-item.layout-grid-item--with-gutter-s-only.grid-col-10-m.grid-col-push-1-m.grid-col-6-xl.grid-col-push-2-xl.article-body--custom-column > section > div.article-body__date-source > time")
            date = [date[0].text if date != [] else ""]
            text = " ".join([i.text for i in soup.select(
                "#content > div:nth-child(7) > div > article > div.article-body > div > div.article-body__section.layout-grid-container.article-body__last-section.article-body__first-section > div.article-body.layout-grid-item.layout-grid-item--with-gutter-s-only.grid-col-10-m.grid-col-push-1-m.grid-col-6-xl.grid-col-push-2-xl.article-body--custom-column > div.article-body__content > p")])
            reporter = soup.select(
                "#content > div:nth-child(7) > div > article > div > div > div.article-body__section.layout-grid-container.article-body__last-section.article-body__first-section > div.article-body.layout-grid-item.layout-grid-item--with-gutter-s-only.grid-col-10-m.grid-col-push-1-m.grid-col-6-xl.grid-col-push-2-xl.article-body--custom-column > section > div.article-inline-byline > span")
            reporter = [reporter[0].text if reporter != [] else ""]

            news = catagory[0] + " --- " + date[0] + " --- " + title[k] + " --- " + text + " --- " + reporter[0]
            print(news)
            f.write(news)

        f.close