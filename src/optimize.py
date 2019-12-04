from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img
import collections

STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
STYLE_LAYERS_SHAPE = [None, None, None, None, None]
STYLE_LAYERS_SIZE = [None, None, None, None, None]
STYLE_LAYERS_INDEX = [-1, -1, -1, -1, -1]
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(content_targets, style_targets, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 

    # style_features = collections.defaultdict()
    style_features = []

    batch_shape = (batch_size,256,256,5)
    style_shape = (1, 256, 256, 3)
    # print(style_shape)

    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
    # with tf.Graph().as_default(), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)

        for i in range(len(style_targets)):
            index = 0
            style_pre = np.array([style_targets[i]])
            # current_style_feature = []
            current_style_feature = np.array([])

            for layer in STYLE_LAYERS:
                # print(layer)
                features = net[layer].eval(feed_dict={style_image:style_pre})
                # print(features.shape)
                features = np.reshape(features, (-1, features.shape[3]))
                # print(features.shape)
                gram = np.matmul(features.T, features) / features.size
                # print(gram.shape)

                if not STYLE_LAYERS_SHAPE[STYLE_LAYERS.index(layer)]:
                    STYLE_LAYERS_SHAPE[STYLE_LAYERS.index(layer)] = gram.shape

                if not STYLE_LAYERS_SIZE[STYLE_LAYERS.index(layer)]:
                    STYLE_LAYERS_SIZE[STYLE_LAYERS.index(layer)] = gram.size

                if STYLE_LAYERS_INDEX[STYLE_LAYERS.index(layer)] == -1:
                    STYLE_LAYERS_INDEX[STYLE_LAYERS.index(layer)] = index

                index = index + gram.size

                # style_features[i][layer] = gram
                # current_style_feature.append(gram.tolist())
                # current_style_feature.append(gram.reshape(-1))
                current_style_feature = np.append(current_style_feature, gram.reshape(-1))

            # style_features.append(np.array(current_style_feature).reshape(-1))
            style_features.append(current_style_feature)

        style_features = np.array(style_features, dtype=np.float32)
        # tf.convert_to_tensor(style_features)

    with tf.Graph().as_default(), tf.Session() as sess:
        lambda_style = tf.placeholder(tf.float32, name="lambda_style")
        style_id = tf.placeholder(tf.int32, name="style_id")
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")

        X_pre = vgg.preprocess(X_content[:,:,:,0:3])

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0)
            preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER]) * batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])

        # content_loss = (1 - lambda_style) * (2 * tf.nn.l2_loss(
        #     net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        # )

        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )  # original

        style_losses = []
        for style_layer in STYLE_LAYERS:
            # print(style_layer)
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size

            # test = lambda_style.eval(session=sess)
            # print('test : ' + test)
            # style_gram = style_features[style_id.eval(session=sess)][style_layer]
            # style_gram = style_features[0][style_layer]

            # s_id = sess.run(style_id)
            # style_gram = style_features[0][style_layer]

            # style_features = [tf.convert_to_tensor(x) for x in style_features]
            # s_gram = tf.gather_nd(style_features, style_id)
            # style_gram = s_gram[style_layer]

            # make style_features into tensor
            # keys = []
            # values = []
            #
            # for k, v in style_features.items():
            #     keys.append(k)
            #     values.append(k)

            # style_features_tf = tf.contrib.lookup.HashTable(
            #     initializer=tf.contrib.lookup.KeyValueTensorInitializer(
            #         keys=tf.constant(range(len(style_targets))),
            #         # values=tf.constant([style_features[i] for i in range(len(style_targets))]),
            #         # values=tf.constant(style_targets),
            #         values=style_targets,
            #     ),
            #     default_value=tf.constant(-1),
            #     name="style_features_tf"
            # )
            # current_style_features = style_features_tf.lookup(style_id)
            # style_gram = tf.gather_nd(current_style_features, STYLE_LAYERS.index(style_layer))
            # style_features_tf.init.run()
            # print(style_gram.eval())
            # print(style_gram.eval())
            style_index = STYLE_LAYERS.index(style_layer)
            style_grams = tf.gather_nd(tf.constant(style_features), [style_id])
            style_gram = style_grams[STYLE_LAYERS_INDEX[style_index]: STYLE_LAYERS_INDEX[style_index] + STYLE_LAYERS_SIZE[style_index]]
            style_gram = tf.reshape(style_gram, STYLE_LAYERS_SHAPE[style_index])
            # style_grams = style_features[style_id]

            # style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/STYLE_LAYERS_SIZE[style_index])

        style_loss = lambda_style * functools.reduce(tf.add, style_losses) / batch_size
        # style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size  # original

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        loss = content_loss + style_loss + tv_loss

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(epochs):
            print('epoch: {}'.format(epoch))
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:
                # print('iterations: {}'.format(iterations))
                start_time = time.time()
                curr = iterations * batch_size

                curr_lambda_style = np.random.randint(1, 100) * 1.0
                curr_lambda_style_img = np.ones((256, 256, 1)) * curr_lambda_style

                curr_style_id = np.random.randint(len(style_targets)) if epoch > 1 else 0
                # print('\ncurr_style_id:')
                # print(type(curr_style_id))
                # print(curr_style_id)
                curr_style_channel = np.ones((256, 256, 1)) * curr_style_id

                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                    try:
                        curr_img = get_img(img_p, (256, 256, 3)).astype(np.float32)
                    except Exception:
                        continue

                    X_batch[j,:,:,0:3] = curr_img
                    X_batch[j,:,:,3:] = curr_lambda_style_img
                    X_batch[j,:,:,4:] = curr_style_channel

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                    X_content:X_batch,
                    lambda_style: curr_lambda_style,
                    style_id: curr_style_id
                }

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))

                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples

                should_print = is_print_iter or is_last

                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch, lambda_style: 80.0, style_id: 0  # np.random.randint(1, 10) / 10.0
                    }

                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    # print(losses)
                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                       saver = tf.train.Saver()
                       res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
