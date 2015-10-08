__author__ = 'simon'
import sys
sys.path.append('/home/simon/PycharmProjects/glass/palovito')
from sql import PGConnector
from googleads import adwords

def unify(obj):
    #print obj
    if isinstance(obj, str):
        return unicode(obj, 'latin')
    elif isinstance(obj, dict):
        nd = {}
        #print obj
        for k, v in obj.iteritems():
            #print "unifying ", k
            nk = unify(k)
            #print "--> ", nk
        #print "unifying ", v
            nv = unify(v)
            #print "--> ", nv
            nd[nk] = nv
        return nd
    elif isinstance(obj, list):
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



AD_GROUP_ID = '21405570899'


def add_advert(client, ad_group_id, details):
  # Initialize appropriate service.
  ad_group_ad_service = client.GetService('AdGroupAdService', version='v201506')

  # Construct operations and add ads.
  # If needed, you could specify an exemption request here, e.g.:
  # 'exemptionRequests': [{
  #     # This comes back in a PolicyViolationError.
  #     'key' {
  #         'policyName': '...',
  #         'violatingText': '...'
  #     }
  # }]
  operations = [
      {
          'operator': 'ADD',
          'operand': {
              'xsi_type': 'AdGroupAd',
              'adGroupId': ad_group_id,
              'ad': {
                  'xsi_type': 'TextAd',
                  'finalUrls': details['finalUrls'],
                  'displayUrl': details['displayUrl'],
                  'description1': details['description1'],
                  'description2': details['description2'],
                  'headline': details['headline']
              },
              # Optional fields.
              'status': details['status']
          }
      }
  ]
  ads = ad_group_ad_service.mutate(operations)

  # Display results.
  for ad in ads['value']:
    print ('Ad with id \'%s\' and of type \'%s\' was added.'
           % (ad['ad']['id'], ad['ad']['Ad.Type']))


if __name__ == '__main__':
    # Initialize client object.
    adwords_client = adwords.AdWordsClient.LoadFromStorage()

    #main(adwords_client, AD_GROUP_ID)

    pgcs = PGConnector({"host":"192.168.2.76", "user":"mick", "password":"mick", "dbname":"backend"})
    res = pgcs.select("""select * from adwords.ad_text where id=6316943""")

    for r in res['data']:
        for i, f in enumerate(r):
            print i, f
        data = r
        details = {'status':r[7], 'finalUrls':r[15], 'displayUrl':r[9], 'description1':r[12], 'description2':r[13], 'headline':r[11]}
        add_advert(adwords_client, AD_GROUP_ID, unify(details))