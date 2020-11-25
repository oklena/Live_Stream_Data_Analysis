# Live Stream Data Analysis
Capstone Project

My project is making predictions of number new followers using machine learning predictive models. Project goal is to help streamers to gain more followers in order to get sponsorships. I believe live streaming is a good way to earn money, since it is becoming more and more popular, especially during this pandemic that we are all experiencing right now. With more people losing their regular jobs, it is great to have live streaming platforms offering opportunity to make money. My project is oriented to individual streamers as well as to sponsors who wants to partner with a streamer to advertise their products. I am using a data from twitch.tv live streaming service that I found on Kaggle.com. Twitch.tv gained popularity with computer games streaming, but lately musicians’ streams are getting popular, also life chats, so you don’t even need to be skillful in something particularly to become a streamer, you can be good at talking and it would enough. Twitch’s parent company is Amazon. The data contains number of viewers, followers and duration of top 1000 streamers. 

[Data](https://www.kaggle.com/aayushmishra1512/twitchdata)

# Exploratory Data Analysis
This graph shows that followers gained reaches pick at around 50000 minutes which is equal to 35 days. A streamer can gain a good amount of new followers in about 1 month with quality content.
![image](https://github.com/oklena/Live_Stream_Data_Analysis/tree/main/reports/figures/StreamTimevsFollowersgained.jpg)
Since it is an American platform, English is the most popular language, but Korean and Russian are getting popular. 
![image](https://github.com/oklena/Live_Stream_Data_Analysis/tree/main/reports/figures/Languages.jpg)
Streamers on twitch.tv have options. They can turn on Mature content if they are targeting adult audience, I am assuming viewers should be older than 18 years old. From the graph you can see that most of the streamers don’t have this option on, but they should be careful with their vocabulary. 
![image](https://github.com/oklena/Live_Stream_Data_Analysis/tree/main/reports/figures/Mature.jpg)
This data on top 1000 streamers, so most of them are partnering with twitch. They pay some percentage of their earnings to twitch.tv.
![image](https://github.com/oklena/Live_Stream_Data_Analysis/tree/main/reports/figures/Partnered.jpg)
Target is exponential.
![image](https://github.com/oklena/Live_Stream_Data_Analysis/tree/main/reports/figures/Target_Dist.jpg)

# Modeling 
![image](https://github.com/oklena/Live_Stream_Data_Analysis/tree/main/reports/figures/Models_performances.jpg)
Poisson Regressor model is predicting correctly ½ time. MAPE (Mean Absolute Percentage Error) is negative, so it is hard to interpret. 5 important features are Watch Time, Folowers, Language, Partnered and Peak viewers.
![image](https://github.com/oklena/Live_Stream_Data_Analysis/tree/main/reports/figures/PR_5_feature_importances.jpg)
![image](https://github.com/oklena/Live_Stream_Data_Analysis/tree/main/reports/figures/Poisson_Model_performances.jpg)

# Conclusion
Longer people watch, most likely they will start following.
Partnering with twitch is beneficial. 
