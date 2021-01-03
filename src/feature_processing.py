import pandas as pd


def feature_pruning(fr,features_req,values_req):
    print("performing feature pruning ......")
    features = []
    values = []
    for pair in fr:
        if features_req == 0 and values_req == 0:
            break;

        if len(pair[1]) > 5:
            if pair[1][-1] == 'J' and values_req > 0:
                values_req -= 1
                values.append(pair[1])
            elif pair[1][-1] == 'N' and features_req > 0:
                features_req -= 1
                features.append(pair[1])
    
    return (features,values)


def feature_extraction(df,features_req = 20,values_req = 20):
    print("extracting features ......")
    dic = {}
    fr = []
    total = 0

    for i in range(0,len(df)):
        text = df.iloc[i,1]
        for word in text.split():
            if word in dic.keys():
                fr[dic[word]][0] += 1;
            else:
                fr.append([1,word])
                dic[word] = len(fr)-1;
            total += 1
    
    for i in range(0,len(fr)):
        fr[i][0] /= total;
    fr = sorted(fr,key=lambda x:x[0],reverse=True)

    features,values = feature_pruning(fr,features_req,values_req)

    # writing the features and values in file.
    try:  
        file1 = open("features_after_prunning.txt","w")
        file1.write("Features : \n")
        for i in features:
            file1.write(i + "\n")
        file1.write("\n\nValues : \n")
        for i in values:
            file1.write(i + "\n")
    except:
        print("! cannot write to the file now")
    
    return features,values



def create_feature_vector(df,features,values):
    print("creating feature vectors .....")
    feature_vectors = [] # feature vectors of all reviews
    y = [] # store the output class
    for i in range(0,len(df)):
        vector = [] # feature vector for ith review
        text = df.iloc[i,1] # text for the ith review
        verdict = df.iloc[i,2]
        pos = 0 # pointer to the position of word
        head = [] # queue for bfs
        adj = {} # adjacency list
        last = None # most recent value to the left
        text = text.split(' ')
        right = []
        for i in range(0,len(text)):
            right.append(None)
        
        for i in range(len(text)-1,-1,-1):
            if text[i] in values or text[i] in features:
                right[i] = text[i]
            else:
                if i != len(text)-1:
                    right[i] = right[i+1]

        for word in text:
            if word in values:
                head.append(word)
            if last != None and (word in values or word in features):
                if word not in adj.keys():
                    adj[word] = [[0,last]]
                else:
                    adj[word].append([0,last])
            if right[pos] != None:
                if word not in adj.keys():
                    adj[word] = [[1,right[pos]]]
                else:
                    adj[word].append([1,right[pos]])
            
            if word in values or word in features:
                last = word
            
            pos += 1
            
        for key in adj.keys():
            adj[key] = sorted(adj[key],key = lambda x:x[0])
        # performing BFS to map features/aspects to the values
        vis = {} # make count of visited nodes i.e words here
        for i in head:
            vis[i] = i
        while(len(head)):
            top = head[0]
            head.pop(0)

            for child in adj[top]:
                if child[1] not in vis:
                    vis[child[1]] = vis[top]
                    head.append(child[1])

        for key in vis.keys():
            if key in features:
                s = str(key)+"_"+str(vis[key])
                vector.append(s)
        
        feature_vectors.append(vector)
        y.append(verdict)
    
    return feature_vectors,y
            

                
           

