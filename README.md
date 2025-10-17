# Customer Purchase Data Analysis

This is an analysis of customers’ transaction data of an online retail store to gain insights about
customers’ purchase behavior, in order to help the business, make effective decisions about
inventory management, monitoring dynamic shopping trends, and channel marketing resources in
the correct direction. The RFM (Recency, Frequency, and Monetary values) analysis was used for
initial analysis and aggregation and using K-Means unsupervised learning algorithm, four clusters,
each with about 20-30% of the total number of customers were identified with distinct purchase
behavior and marketing needs.

**Solution Methodology:** 
The analysis approach followed to create customer segments using the online shopping data is
summarized in the flow chart below.
<img width="481" height="558" alt="Customer segmentation method image" src="https://github.com/user-attachments/assets/962a43ab-512f-4f5d-92bf-2954c4d3bf47" />

**Customer Segmentation using unsupervised learning**
Among the various algorithms considered here, K-Means clustering was chosen as it was the best suited for the available dataset
since it is simple, fast and works well of large datasets.

Visualization of clustering:
A 3D plot was generated with data points for each customer using R, F, M values on the axes.
As seen in the plot the clusters are color coded and show a clear distinction in 3D space. The
centroids of clusters formed by K-Means are also shows in the plot as black dots
<img width="587" height="556" alt="3D Viz of customer segments" src="https://github.com/user-attachments/assets/93022fd2-6f52-422a-9ea6-06944f5196f3" />

This gives valuable insights into the shopping behavior of the customers in each cluster:
• Cluster 0 (31.44%): Very high R, coupled with very low F and M shows that these are
recent new customers who are just trying out the products since they are spending small
amounts and are not buying repeatedly yet. This segment may be a result of some new
product introduced in the website or a result or marketing campaigns to attract new
customers. It is interesting to see that this is the largest of the four segments, so marketing
resources should focus on earning the trust and repeated business of this segment for future
growth.
• Cluster 1 (20.74%): Low values of R, F and M show the customers spent very little a long
time ago and did not return to buy anything else. This is the segment of customers who are32
almost lost and very not very profitable to begin with. Marketing resources should not be
spent on this segment.
• Cluster 2 (20.35%): Very high values of F and M, with low value of R means shows that
this segment of customers very frequent buyers on the e-commerce website and also very
big spenders. However, they have not made very recent transactions. This segment,
although the smallest of the four segments, should be the highest priority to focus on for
marketing because the website seems to be losing highly profitable customers.
• Cluster 3 (27.45%): Average values all across R, F and M shows that these are regular
customers who shop frequently and spend moderately and continue to shop at the website.
This segment is loyal to the e-commerce website and the target should be to get more
customers to this segment

**Analysis and Results**
RFM analysis assigns a score of 1-4 for R, F and M attributes of customers, with 1 being the least
desirable and 4 being the most desirable. Using the RFM scores, customer shopping behavior can
be inferred and marketing recommendations are suggested for different combinations of R, F and
M values as seen in Table below.

<img width="600" height="279" alt="RFM segments table" src="https://github.com/user-attachments/assets/9e4e14e6-61d2-4566-ab64-6997c086bde9" />

**Conclusions**
The online retail store data was used and extensively analyzed using RFM analysis. 
An unsupervised learning classification algorithm called K-Means was also used to create four customer segments. It was seen that the customer segmentation created by K-Means was of high quality in the sense that it created four distinct categories with clearly distinct shopping behaviors. The segments are fairly sized (between 20-30% of total customers each) making each of the segments significant. It would be very time consuming to come up with these customer segments manually using RFM analysis. The customer segments were further analyzed and
marketing strategies were developed for each category.

