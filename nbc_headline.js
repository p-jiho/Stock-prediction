var client = require("cheerio-httpcli");

var url = "https://www.nbcnews.com/archive/articles/2022/january";
var param = {};

var n = 0;
client.fetch(url, param, function(err, $, res){
    if (err) {console.log("error"+err); return;}

    $("body > div.Layout > main > a").each(function(idx){ 

        var text = $(this).text();
        n = n+1;
        console.log(text);
    });
    console.log(n);
});
