__author__ = 'simon'

from sql import PGConnector

def createCategoryTree():
    pass

def assignProductsToTree(tree, products):
    pass

def getCategoryProducts(pgc, catId):
    res = pgc.select("""select asin from clothes.amazon_products where browse_node_id=%s and status='active'""", (catId,))
    products = [r[0] for r in res['data']]
    return products

def getTopCategories(pgc):
    res = pgc.select("""select c.display_name, c.id, count(*) from clothes.amazon_products p, clothes.amazon_categories c
                         where c.id=p.browse_node_id and p.status='active' group by c.display_name, c.id order by 3 desc""")
    return res['data']

if __name__=="__main__":
    cats=["Women's Dresses","Men's T-Shirts","Men's Shirts","Men's Jackets","Women's Tops",
          "Women's Everyday Bras","Women's T-Shirts","Women's Blouses & Shirts"]

    pgcs = PGConnector({"host":"quackdb.cozdd0etxhab.eu-west-1.rds.amazonaws.com", "user":"mick", "password":"crewe2wba0", "dbname":"uk"})
    res = pgcs.select("""select c.display_name, c.id, count(*) from clothes.amazon_products p, clothes.amazon_categories c
                         where c.id=p.browse_node_id and p.status='active' group by c.display_name, c.id order by 3 desc""")

    print res
    for r in res['data']:
        if r[0] in cats and r[2]>2000:
            print r
            products = getCategoryProducts(pgcs, r[1])
            print "%d products" % len(products)
            print "testing to see if images exist..."
            cnt=0
            for p in products:
                try:
                    f = open("/home/simon/data/palovito/images/"+p['asin']+".jpg")
                    cnt+=1
                except:
                    print "image doesnt exist"
            print cnt