import tensorflow as tf
import os

from data_prepare import data_loader
from model_design import model_design_nn
from model_design import calc_loss
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # init directory
    if not os.path.exists(config.model_save_dir):
        os.mkdir(config.model_save_dir)
    if not os.path.exists(config.summary_save_dir):
        os.mkdir(config.summary_save_dir)

    # data_load
    Data_Gen = data_loader.DataGenerator(img_dir=config.img_dir,
                                         img_h=config.img_h,
                                         img_w=config.img_w,
                                         img_ch=config.img_ch,
                                         batch_size=config.batch_size)
    Data_Gen.build_data()
    generator = Data_Gen.next_batch()


    # define train model
    images = tf.placeholder(dtype=tf.float32, shape=[None, config.img_h, config.img_w, config.img_ch])
    labels = tf.placeholder(dtype=tf.int32, shape=[None])
    logits = model_design_nn.cls_net(inputs=images, is_training=True)
    cls_loss = calc_loss.build_loss(logits=logits, labels=labels)

    # l2_loss
    l2_loss = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)
    for scope_name in ['CNN_Module', 'FC_Module']:
        module_train_vars = tf.trainable_variables(scope=scope_name)
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(var) for var in module_train_vars])
        l2_loss += regularization_cost * config.l2_loss_lambda
    loss = cls_loss + l2_loss

    tf.summary.scalar("Loss/0_total_loss", loss)
    tf.summary.scalar("Loss/1_cls_loss", cls_loss)
    tf.summary.scalar("Loss/2_l2_loss", l2_loss)

    # summary_op
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(config.summary_save_dir)

    # train_op
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss=loss, global_step=global_step)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=3)
    ckpt_path = tf.train.latest_checkpoint(config.model_save_dir)
    print('latest_checkpoint_path: ', ckpt_path)
    if ckpt_path is not None:
        saver.restore(sess, ckpt_path)
        prev_step = int(ckpt_path.split('-')[-1])
    else:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        prev_step = -1

    # train
    with sess.as_default():
        for i in range(config.train_steps):
            _inputs, _outputs = next(generator)  # .__next__()
            _img_tensor = _inputs['images']
            _label_tensor = _outputs['labels']
            # print(_ant_tensor.shape)
            _loss, _, _summary_op = sess.run([loss, train_op, summary_op],
                                             feed_dict={
                                                 images: _img_tensor,
                                                 labels: _label_tensor
                                             })
            print('step: ', prev_step + 1 + i, 'loss: ', _loss)

            train_writer.add_summary(_summary_op, prev_step + 1 + i)
            train_writer.flush()

            if i % config.save_n_iters == 0:
                saver.save(sess=sess,
                           save_path=os.path.join(config.model_save_dir, 'model.ckpt'),
                           global_step=global_step)







if __name__ == '__main__':
    main()





