#encoding ='utf-8'
from __future__ import division
import re
import jieba
from gensim import corpora, models, similarities
import pandas as pd


class DataProcessor(object):
    
    def __init__(self, train, test):
        self.train = train 
        self.test = test
     
    ## preprocess the data, including remove the null data, add the column name
    def preprocess(self):
        ### check the info
        train = self.train
        ## drop null data
        train.dropna(inplace=True)
        ## rename the columns 
        train.rename(columns={0: 'Product Name', 1: 'Category', 2:'Query', 3:'Event',4:'Date' }, inplace=True)
        ### drop the row including the repeated column name
        train.drop(train.index[train[train.Category =='Category'].index],inplace=True)
        return train
              
    ### to segment the product name    
    def jieba_fenci(self, data):
        regex = re.compile(ur"[^\u4e00-\u9f5aa-zA-Z0-9]") # keep english,chinese character and number
        fenci = map(lambda x : '|'.join(jieba.cut(regex.sub(' ',x))),data['Product Name'])
        return fenci
    
    ### segment the query, to get the relevance between the query and the new product
    def query_seg(self, query):
        query= unicode(query)
        query = query.strip()
        query_seg = '|'.join(jieba.cut(query)).split('|')
        return query_seg 

    ### remove the null data after the segment
    def remove_space(self,data):
        while ' ' in data:
            data.remove(' ')
        return data
    
    def lower_case(slef, data):
        data = map(lambda x: x.lower(),data)
        return data
    
    #### build a similarity index model using tfidf model
    def get_index(self):
        ### create a dictionary
        train = self.train
        texts = []
        for ix, row in train.iterrows():
            texts_cuts = row['Product Seg']
            texts.append(texts_cuts)
        #from texts to extract bag of words, the dictionary is comprised of # unique tokens  
        self.dictionary = corpora.Dictionary(texts)
        print 'generating the dictionary'
        print self.dictionary
        ## print dictionary.token2id
        ## corpus: the document is representd with id and the term frequency
        corpus = [self.dictionary.doc2bow(text) for text in texts] 
        ##print corpus
        ### using the corpus to train a tfidf model
        self.tfidf = models.TfidfModel(corpus) #initialize a model
        print 'tfidf model info:', self.tfidf 
        self.corpus_tfidf = self.tfidf[corpus]  #convert the corpus to vectors in the tfidf space.
        ### compute the len of the vector in the courpus-tfidf
        total_len = map(lambda x: len(x),self.corpus_tfidf)
        print 'the maximum len of the vector', max(total_len)
        print 'the average len of the vector', sum(total_len)/float(len(self.corpus_tfidf))
        
        self.index = similarities.MatrixSimilarity(self.corpus_tfidf) 
        return self.index
    
       ### 'Impression'/'Click' count for the query
    def event_count(self, event,query_term_all):
        count_number = []
        new_event = self.train[self.train['Event'] == event]
        query_term_event = list(new_event['Query'].values)
        for item in query_term_all: 
            if item in query_term_event:
                count_number.append(query_term_event.count(item))
            else: 
                count_number.append(0)
        return  count_number
    
     ### get a lsi model   
    def get_lsi(self):
        self.lsi_model =models.LsiModel(self.corpus_tfidf,id2word = self.dictionary, num_topics = 1000)
        return self.lsi_model
    
    ### using similarity index to get the similar product list
    #### get  a similar product list with the lengh of num_sim for a newly input product
    def get_similar_product_list(self,product_texts_cuts, num_sim,column_list):
        ## convert the strings to vector space using the dictionary generated from the training corpus
        vec_bow = self.dictionary.doc2bow(product_texts_cuts) 
        #print vec_bow
        ## convert the vector-space representations of the strings to IF-IDF space
        vec_tfidf = self.tfidf[vec_bow] 
        #get the similarith between the current input product and all the products from sample
        sims = self.index[vec_tfidf] 
        sort_sims = sorted(enumerate(sims), key = lambda item:-item[1]) # sort the similarity in a ascending order
        #store the index, and the similarity in a list
        index_list = []
        sim_list = []
        for index,sim in sort_sims[0:num_sim]:
            index_list.append(index)
            sim_list.append(sim)
        ### retrieve the product info from the train data  
        recommend_list = self.train.iloc[index_list][column_list]
        recommend_list ['similarity'] = sim_list 
        return recommend_list 
    
  
    ### using lsi model to get the similar product list
    #### get the most similar product list for a newly input product
    def get_similarity_product_list_lsi(self,product_texts_cuts, num_sim,column_list):      
        vec_bow = self.dictionary.doc2bow(product_texts_cuts) 
    #print vec_bow
        vec_tfidf = self.tfidf[vec_bow] #get the tfidf value for the current product
        ## get the lsi model
        lsi_model =self.get_lsi()
        corpus_lsi =lsi_model[self.corpus_tfidf]
        test_lsi = lsi_model[vec_tfidf]
        corpus_simi_matrix = similarities.MatrixSimilarity(corpus_lsi) 
        test_simi= corpus_simi_matrix[test_lsi]
        sort_sims = sorted(enumerate(test_simi), key = lambda item:-item[1]) # sort the similarity in a ascending order
        #store the index, and the similarity in a list
        index_list = []
        sim_list = []
        for index,sim in sort_sims[0:num_sim]:
            index_list.append(index)
            sim_list.append(sim)
        ### retrieve the product info from the train data  
        recommend_list = self.train.iloc[index_list][column_list]
        recommend_list ['similarity'] = sim_list 
        return recommend_list 
 
    ### this function is to find the relevance between the recommended query and the new product
    def get_relevance(self,recommend_query, product_texts_cuts,row_location):
        relevance = [] 
        for value_query in recommend_query:
            #print value_query
            cleaned_query = self.remove_space(self.query_seg (value_query))
            len_query = len(cleaned_query)
            relevance_item_count = 0
            for query_item in cleaned_query: 
                #print query_item
                if query_item in self.test['Product Name'][row_location]:
                    relevance_item_count = relevance_item_count + 1
                    #print 'relevant'
                elif query_item in product_texts_cuts:
                    relevance_item_count = relevance_item_count + 1
                    #print 'relevant here'
            relevance_score = relevance_item_count/len_query
            #print 'relevance_score ',relevance_score 
            relevance.append(relevance_score ) 
        return relevance
    

    #### for a similar product list, to calculate more information 
    ### get all the eavluator = [query_count,ctr,similrity,click,relevance,event]
    def get_evaluate_product_list(self,weight,recommend_list,product_texts_cuts,row):
        
        event_dummies = pd.get_dummies(recommend_list['Event'],prefix ='Event')
        recommend_list = pd.concat([recommend_list,event_dummies],axis=1)
        ### ge the revelance score
        query_list = list(recommend_list['Query'].values)
        recommend_list ['query_count'] = map(lambda x : query_list.count(x), query_list)
        ### ge the revelance score
        recommend_list['relevance_score']= self.get_relevance(recommend_list['Query'], product_texts_cuts,row)

        ### weights for each evaluator
        #weight = [0.1, 0.1, 0.1,0.1, 0.3, 0.3] ## [query_count,ctr,similrity,click,relevance,event]
        query_count = recommend_list['query_count']/recommend_list['query_count'].max()*weight[0]  
        ctr = recommend_list['CTR']*weight[1]  
        similrity = recommend_list['similarity']*weight[2]  
        click = recommend_list['Click No']/recommend_list['Click No'].max()*weight[3]   
        relevance = recommend_list['relevance_score']*weight[4]  
        event = recommend_list['Event_Click']*weight[5]
        recommend_list['total_score'] = query_count+ctr +similrity+click+relevance+event      
        recommend_list.sort_values('total_score',ascending=False,inplace =True)
        return recommend_list
    
    
    ### recommend kewwords based on the performance of the query
    def get_recommend_keyword(self,recommend_list,number_keyword,row):
        keyword_list = []
        for value in recommend_list[ ['Query','total_score']].values:
            if len(keyword_list) < number_keyword:
                if (value[1]!=0) and (value[0] not in keyword_list):
                    keyword_list.append(value[0])
        if len(keyword_list) ==0:
            for value in recommend_list[ ['Query','total_score']].values:
                if len(keyword_list) < number_keyword:
                    if value[0] not in keyword_list:
                        keyword_list.append(value[0])
        ### output the recommended keywords
        self.test['keyword recommend'][row] = ','.join(keyword_list)
        return self.test
    
    
    

            
    