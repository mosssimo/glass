__author__ = 'simon'

import re
import random
import psycopg2 as psycopg

def getPostgresConnection(params):
    str = ' '.join(['%s=%s'%(k,v) for k,v in params.items()])
    conn = psycopg.connect(str)
    return conn

def getMainConnection():
    conn = getPostgresConnection({'host':'db.production.frontend.pricegoblin.co.uk','user':'django','password':'django','dbname':'pricegoblin'})
    return conn

def getBackendConnection():
    conn = getPostgresConnection({'host':'192.168.2.76','user':'mick','password':'mick','dbname':'backend'})
    return conn

def getLocalConnection():
    conn = getPostgresConnection({'host':'localhost','user':'postgres','dbname':'postgres'})
    conn.set_client_encoding('utf8')
    return conn


def unify(obj):
    #print obj
    if isinstance(obj,str):
        return unicode(obj,'latin1')
    elif isinstance(obj,dict):
        nd = {}
        #print obj
        for k,v in obj.iteritems():
            #print "unifying ", k
            nk = unify(k)
            #print "--> ", nk
            #print "unifying ", v
            nv = unify(v)
            #print "--> ", nv
            nd[nk] = nv
        return nd
    elif isinstance(obj,list):
        nl = []
        for item in obj:
            nl.append(unify(item))
        return nl
    elif isinstance(obj, tuple):
        nl = []
        for item in obj:
            nl.append(unify(item))
        return tuple(nl)
    else:
        return obj

moneyRegex = re.compile('\xa3(\d+\.\d*)')


def pgInsert(conn, qstr, params=None):
    try:
        curs = conn.cursor()
        if params is None:
            curs.execute(qstr)
        else:
            curs.execute(qstr, params)
        r = {"status":"ok"}
    except Exception, ex:
        #print "Exception ", ex
        r = {"status":"error", "error":"%s"%str(ex)}
    except:
        #print "UNknown exception"
        r = {"status":"error", "error":"Unknown"}
    finally:
        curs.close()
        conn.commit()
    return r

def pgUpdate(conn, qstr, params=None):
    try:
        curs = conn.cursor()
        if params is None:
            c = curs.execute(qstr)
        else:
            c = curs.execute(qstr, params)
        #row = curs.fetchone()
        sm = curs.statusmessage.split()
        if sm[-1]=='0':
            print 'Update error'
            print 'SQL : ', qstr
            print params
            print sm
            return {"status":"error","error":"no data entered during update. (Table exists?)", "data":c}

        conn.commit()
        r = {"status":"ok","data":sm[-1]}
    except Exception, ex:
        #print "Exception ", ex
        r = {"status":"error", "error":"%s"%str(ex)}
    except:
        #print "UNknown exception"
        r = {"status":"error", "error":"Unknown"}
    finally:
        curs.close()
        conn.commit()
    return r

def pgSelect(conn, qstr, params=None):
    try:
        curs = conn.cursor()
        if params is None:
            curs.execute(qstr)
        else:
            curs.execute(qstr, params)
        rows = curs.fetchall()
        desc = curs.description

        r = {"status":"ok", "data":rows, "columns":desc}

    except Exception, ex:
        r = {"status":"error", "error":"%s"%str(ex)}
    except:
        r = {"status":"error", "error":"Unknown"}
    finally:
        curs.close()

    return r

def pgDelete(conn, qstr, params=None):
    try:
        curs = conn.cursor()
        if params is None:
            curs.execute(qstr)
        else:
            curs.execute(qstr, params)
        conn.commit()
        r = {"status":"ok"}
    except Exception, ex:
        #print "Exception ", ex
        r = {"status":"error", "error":"%s"%str(ex)}
    except:
        #print "UNknown exception"
        r = {"status":"error", "error":"Unknown"}
    finally:
        curs.close()

    return r


class PGConnector(object):
    def __init__(self, details):
        self.details = details
        self.conns = []

    def getNewConnection(self):
        conn = getPostgresConnection(self.details)
        return conn

    def getConnection(self):
        if len(self.conns)==0:
            conn = self.getNewConnection()
            return conn
        else:
            conn = self.conns.pop()
            return conn

    def select(self, qstr, params=None):
        conn = self.getConnection()
        res = pgSelect(conn, qstr, params)
        self.checkInConnection(conn)
        return res

    def insert(self, qstr, params=None):
        conn = self.getConnection()
        res = pgInsert(conn, qstr, params)
        self.checkInConnection(conn)
        return res

    def delete(self, qstr, params=None):
        conn = self.getConnection()
        res = pgDelete(conn, qstr, params)
        self.checkInConnection(conn)
        return res

    def update(self, qstr, params=None):
        conn = self.getConnection()
        res = pgUpdate(conn, qstr, params)
        self.checkInConnection(conn)
        return res

    def checkInConnection(self, conn):
        self.conns.append(conn)

    def closeAll(self):
        for c in self.conns:
            c.close()

if __name__=="__main__":
    pgcs = PGConnector({"host":"quackdb.cozdd0etxhab.eu-west-1.rds.amazonaws.com", "user":"mick", "password":"crewe2wba0", "dbname":"uk"})
    res = pgcs.select("""select c.display_name, c.id, count(*) from clothes.amazon_products p, clothes.amazon_categories c
                         where c.id=p.browse_node_id and p.status='active' group by c.display_name, c.id order by 3 desc""")

    for r in res['data']:
        print r