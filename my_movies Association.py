# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 21:26:50 2020

@author: Lenovo
"""

movies = []
# loading file in transaction format
with open("my_movies.csv") as f:
    movies = f.read()

#splitting the data "\n"
movies = movies.split("\n")
movies_list = []
for i in movies:
    movies_list.append(i.split(","))    
all_movies_list = []
for i in movies_list:
    all_movies_list = all_movies_list+i

movie_freq = Counter(all_movies_list)

movie_freq = sorted(movie_freq.items(),key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in movie_freq]))
items = list(reversed([i[0] for i in movie_freq]))

plt.bar(height = frequencies[0:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[0:11]);plt.xlabel("items");plt.ylabel("Count")

movies_series  = pd.DataFrame(pd.Series(movies_list))

movies_series = movies_series.iloc[:11,:]

movies_series.columns=['movies1']

X = movies_series['movies1'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)

frequent_itemsets.sort_values('support',ascending = False,inplace=True)

plt.bar(x = list(range(0,11)),height = frequent_itemsets.support[0:11],color='rgmyk');plt.xticks(list(range(0,11)),frequent_itemsets.itemsets[0:11]);plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values('lift',ascending = False,inplace=True)

#To eliminate Redudancy in Rules 

def to_list(i):
    return (sorted(list(i)))
#Sorting, listing and appending

ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
# getting rules without any redudancy 

rules_no_redudancy  = rules.iloc[index_rules,:]
# Sorting them with respect to list and getting top 10 rules 

rules_no_redudancy.sort_values('lift',ascending=False).head(10)

plt.bar(x = list(range(0,11)),height = rules_no_redudancy.lift[0:11],color='rgmyk');plt.xticks(list(range(0,11)),rules_no_redudancy.antecedents[0:11])

plt.scatter(rules_no_redudancy['support'],rules_no_redudancy['lift'], alpha=0.5);plt.xlabel('support');plt.ylabel('lift');plt.title('Support vs Lift')
plt.plot(rules_no_redudancy['lift'], rules_no_redudancy['confidence'],'go')