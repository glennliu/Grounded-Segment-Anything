import os, json 

def read_category(category_file):
    mapper = {} # openset name to nyu name
    composite_pairs = {} # pair of openset names, {pair1:pair2}
    
    root_category = {}  # Each nyu object has a list of openset objects
    with open(category_file, 'r') as f:
        data = json.load(f)
        objects = data['objects']
        for rootname,data in objects.items():
            openset_obj = data["main"]
            for ele in openset_obj:
                mapper[ele] = rootname    
            root_category[rootname] = openset_obj
            if 'composite' in data:
                for pair in data['composite']:
                    composite_pairs[pair[0]] = [pair[1],rootname]
                    
            
        f.close()
        return root_category, mapper, composite_pairs
    
if __name__=='__main__':
    
    category_file = '/home/cliuci/code_ws/Grounded-Segment-Anything/categories.json'
    root_category, mapper, composite_pairs = read_category(category_file)
    
    test_name = 'dresser'
    if test_name in mapper:
        print('find {}->{}'.format(test_name,mapper[test_name]))
    else:
        print('ignore')
        
    if test_name in composite_pairs:
        print('checking pair ({},{})->'.format(test_name,composite_pairs[test_name][0]),composite_pairs[test_name][1])
    