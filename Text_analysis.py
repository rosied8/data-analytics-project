file.info()
desription=file["a"]
stoplist={'00', '000', '01', '010', '02', '03', '04', '040', '05', '06', '07', '08', '09', '10', '100', '101', '102', '103', '105', '106', '107', '108', '109', '11', '110', '111', '112', '118', '12', '121', '122', '123', '125', '126', '127', '128', '129', '13', '130', '14', '140', '15', '152', '155', '16', '17', '170', '173', '174', '175', '177', '178', '18', '180', '19', '190', '20', '200', '21', '210', '22', '220', '23', '230', '24', '240', '25', '250', '26', '260', '27', '270', '28', '280', '284', '29', '290', '291', '292', '30', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '31', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '32', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '33', '330', '331', '332', '333', '334', '335', '336', '338', '339', '34', '340', '341', '342', '343', '344', '345', '347', '35', '350', '351', '352', '353', '354', '355', '356', '357', '359', '36', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '37', '370', '372', '373', '374', '375', '378', '38', '380', '383', '385', '388', '389', '39', '390', '399', '40', '400', '402', '408', '41', '410', '42', '420', '423', '43', '44', '440', '45', '450', '459', '46', '47', '470', '48', '480', '49', '50', '500', '51', '510', '52', '53', '54', '540', '55', '56', '57', '58', '59', '60', '600', '61', '610', '62', '63', '64', '640', '65', '66', '67', '68', '69', '70', '700', '71', '710', '72', '73', '74', '75', '76', '77', '78', '79', '80', '800', '81', '810', '82', '83', '84', '85', '86', '87', '88', '89', '90', '900', '91', '910', '92', '93', '94', '95', '96', '97', '98', '99'}
#1.count frequency word
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizerc = TfidfVectorizer(min_df =6,stop_words=stoplist)
C = vectorizerc.fit_transform(desription)
feature_con = vectorizerc.get_feature_names()
print(feature_con)
#create a wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,8))
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
wordcloud = WordCloud(background_color="white", max_words=50,font_path='./fonts/simhei.ttf',width=800, height=400).generate(str(feature_con))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()
# count the frequency of top five words, and represent in a bar chart
word_count={}
for i in range(len(desription)):
    comments=desription.iloc[i].split(";")
    for comment in comments:
        if comment in word_count.keys():
            word_count[comment]+=1
        else:
            word_count[comment]=1
sorted_x = sorted(word_count.items(),key=lambda kv: kv[1])
sorted_x=sorted_x[::-1][0:5]
sorted_x=dict(sorted_x)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
#tick_spacing = 1
tick_spacing = 2
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig=plt.figure(figsize=(25,15))
axes=fig.add_subplot(1,1,1)
axes.set_xticks(np.arange(len(sorted_x)))
axes.set_xticklabels(sorted_x.keys())
axes.bar(sorted_x.keys(),sorted_x.values())
fig.tight_layout()
#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.show()