# Deployed Flask implementation of K-RMS Algorithm in Heroku [![HitCount](http://hits.dwyl.com/garain/k-rms-clustering-app.svg)](http://hits.dwyl.com/garain/k-rms-clustering-app)

This is the web app created as an implementation of the publication "K-RMS Algorithm" that uses **Flask** and **Gunicorn** for deploying.


If we want to deploy our project to Heroku, we need a **Web Server Gateway Interface** (WSGI) such as **Gunicorn**.

# App link
[Link to app](https://garain.vision/Authentication/clusteringKRMS)

# Usage of the app
By default if any file with wrong file format is provided, the results shown as output are calculated on the Iris dataset.

Specificatons of the file:

1) 1st row should have data headings, i.e., column headings.
2) The last column should contain the corresponding data labels in an ascending order starting from index 1 (N.B. The number of clusters is calculated using this column.
3) File should be strictly in .csv format.
4) File size should be <=600 KB.

# Output
Dictionary with Accuracy and Clusters centroids alongwith a message showing number of iterations and lowest and highest errors.

# Publication details

# DOI
https://doi.org/10.1016/j.procs.2020.03.188

# Authors
Avishek Garain, Dipankar Das

# Publication date
2020/1/1

# Journal
Procedia Computer Science

# Volume
167

# Pages
113-120

# Publisher
Elsevier

# Abstract
Clustering is an unsupervised learning problem in the domain of machine learning and data science, where information about data instances may or may not be given. K-Means algorithm is one such clustering algorithms, the use of which is widespread. But, at the same time K-Means suffers from a few disadvantages such as low accuracy and high number of iterations. In order to rectify such problems, a modified K-Means algorithm has been demonstrated, named as K-RMS clustering algorithm in the present work. The modifications have been done so that the accuracy increases albeit with less number of iterations and specially performs well for decimal data compared to K-Means. The modified algorithm has been tested on 12 datasets obtained from UCI web archive, and the results gathered are very promising.

# Bibtex

@article{garain2020k,
  title={K-RMS Algorithm},
  author={Garain, Avishek and Das, Dipankar},
  journal={Procedia Computer Science},
  volume={167},
  pages={113--120},
  year={2020},
  publisher={Elsevier}
}

# Harvard-style
Garain, A. and Das, D., 2020. K-RMS Algorithm. Procedia Computer Science, 167, pp.113-120.
