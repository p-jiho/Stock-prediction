// 코드 참고 : https://navydoc.tistory.com/36?category=857759
// 코드 참고 : https://stackoverflow.com/questions/53940043/unhandledpromiserejectionwarning-this-error-originated-either-by-throwing-insid

const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');

let $href = [];
let headline = [];
let year = "2021";
let month = "november";

axios.get("https://www.nbcnews.com/archive/articles/"+ String(year) + "/" + String(month))
    .then(res => {
        const $ = cheerio.load(res.data);
        $("body > div.Layout > main > a").each((index, item)=>{
            $href.push(item.attribs.href)
            const headline_each = $(item).text();
            headline.push(headline_each);
        });
       fs.open("C:/Users/default.DESKTOP-IT64657/Desktop/nbc/" +year + month + ".txt",'w',function(err,fd){ if (err) throw err; console.log('file open complete'); });

        function contentLoad(URL, hl) {
            return new Promise((resolve, reject)=> {
                try {
                axios.get(`${URL}`)
                    .then(res => {
                        const $ = cheerio.load(res.data);
                        const catagory = $("#content > div:nth-child(7) > div > div > section > div.article-hero__bg-container > header > aside > div > a > span").text();
                        const date = $("#content > div:nth-child(7) > div > div > article > div > div.article-body__section.layout-grid-container.article-body__last-section > div.article-body.layout-grid-item.layout-grid-item--with-gutter-s-only.grid-col-10-m.grid-col-push-1-m.grid-col-6-xl.grid-col-push-2-xl.article-body--custom-column > section > div.article-body__date-source > time").text();
                        console.log(catagory + " --- " + date + " --- " + hl);
                        fs.appendFile("C:/Users/default.DESKTOP-IT64657/Desktop/nbc/" +year + month + ".txt", catagory + " --- " + date + " --- " + hl+ "\n", function (err) { if (err) throw err;  });
                    })
                    .then(() => {
                        resolve();
                    });
                } catch (error) {
                    return res.send(error.message);
                  }
                setTimeout(() => reject('woops'), 500);
            })
        }
        
        (async () => {
            for (let i = 0; i < $href.length; i++){
                await contentLoad($href[i], headline[i]);
            }
        })();
    });
