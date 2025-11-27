---
layout: post
title: "Slowfast Network"
date: 2024-01-06
description: Notes on Slowfast Network
tags: [machine-learning]
categories: [Machine Learning]
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

# SLOWFAST NETWORK
- Involves:
        (i) a Slow pathway, operating at low frame rates to capture spatial semantics
        (ii) a Fast pathway can be made very lightweight by reducing it's channel capacity, yet can learn useful temporal information for video recognition.
- Introduction:
        - It is customary in the recognition of images I(x,y) to treat the 2 spatial dimensions x and y symmetrically. 
            
        - Video signals I(x,y,t). Motion is the spatiotemporal counterpart of orientation, but all spatiotemporal orientations are not equally likely. 

        - Slow motions are more likely than fast motions and this has been exploited in Bayesian accounts of how humans perceive motion stimuli. 

        - If all spatiotemporal orientations are not equally likely, then there is no reason for us to treat space and time symmetrically,as it is implicit in approaches to video recognitions based n spatiotemporal convolutions. We might instead 'factor' the architecture to treat spatial structures and temporal events separately.