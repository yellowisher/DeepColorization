# DeepColorization

Just another copy of [interactive-deep-colorization](https://github.com/yellowisher/DeepColorization). You can find decent pytorch code and paper there.  
As a purpose of study, I wrote tensorflow version of it.  
I want to write down what I learned and this might be helpful for someone in someday.  

Actually I didn't run full training and just checked "model basically works".  
And there are a lot of things to be changed. These are just for reference :p

# Note
Here are some things I learned and I changed.
## Batch normalization layer  
In original model, they used <code>CNN -> ReLU -> BatchNorm</code>  
But after some [reading](https://www.quora.com/What-is-the-order-of-using-batch-normalization-Is-it-before-or-after-activation-function), <code>CNN -> BatchNorm -> ReLU</code> makes more sense to me so I tweaked it.  
And because of that, I don't need to add bias at the end of layers. (because BatchNormLayer includes it)

## Shortcut connection
U-net, which is basic model of original network, using concatenation for shortcut connection.  
But they used <code>conv+add</code> rather than <code>concat</code>  
I just noticed they are equivalent; you can find some more explanation [here](https://github.com/junyanz/interactive-deep-colorization/issues/55).

## X110?
We allocates one byte for each LAB color elements(L,A,B). <code>(range of -128 ~ 127)</code>  
But you can find they multiplying result<code>(which is range of -1.0 ~ 1.0)</code> by 110 rather than 128.  
The reason of that is out of that range is considered as "imaginary part". You can find [answer](https://stackoverflow.com/questions/19099063/what-are-the-ranges-of-coordinates-in-the-cielab-color-space) about it.

# TODO
Seperate model and training code with save/load feature.  
Write preprocessing code for input images. (like TFRecord?)
