import yaml
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import Generator, Discriminator


def load_data(cfg):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # [0,255] -> [0,1] -> [-1,1]
    x_train = (x_train/255.) * 2. - 1.

    x_train = np.expand_dims(x_train, axis=3)
    x_train = tf.cast(x_train, dtype=tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices(x_train)
    train_ds = train_ds.shuffle(1000).batch(cfg['batch_size'])
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds


def train(cfg, train_ds):

    # models
    net_g, net_d = Generator(), Discriminator()

    # loss function
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, label_smoothing=cfg['label_smoothing'])

    # optimizers
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['g_lr'])
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['d_lr'])

    # metrics
    disc_loss_tracker = tf.keras.metrics.Mean(name='disc_loss')
    gen_loss_tracker = tf.keras.metrics.Mean(name='gen_loss')

    # tensorboard
    summary_writer = tf.summary.create_file_writer(
        cfg['log_dir']+'/'+cfg['experiment_name'])

    latent_code_size = cfg['latent_code_size']

    # fix latent code to track improvement
    latent_code4visualization = tf.random.normal(shape=(25, latent_code_size))

    for epoch in range(cfg['epochs']):

        for _, real_imgs in train_ds.enumerate():
            
            # PART 1: DISC TRAINING, fixed generator
            latent_code = tf.random.normal(shape=(real_imgs.shape[0],
                                                  latent_code_size))

            with tf.GradientTape() as disc_tape:
                # generate fake images
                generated_imgs = net_g(latent_code)

                # forward pass real and fake images
                real_preds = net_d(real_imgs)
                fake_preds = net_d(generated_imgs)

                y_pred = tf.concat([real_preds, fake_preds], axis=0)
                y_true = tf.concat([tf.ones_like(real_preds),
                                    tf.zeros_like(fake_preds)], axis=0)
                
                # compute loss
                disc_loss = loss_fn(y_true=y_true, y_pred=y_pred)

            # compute disc gradients
            disc_gradients = disc_tape.gradient(disc_loss,
                                                net_d.trainable_variables)

            # update disc weights
            disc_optimizer.apply_gradients(zip(disc_gradients,
                                               net_d.trainable_variables))

            # update disc metrics
            disc_loss_tracker.update_state(disc_loss)

            # PART 2: GEN TRAINING, fixed discriminator
            latent_code = tf.random.normal(shape=(real_imgs.shape[0],
                                                  latent_code_size))

            with tf.GradientTape() as gen_tape:
                # generate fake images
                generated_imgs = net_g(latent_code)

                # forward pass only images
                fake_preds = net_d(generated_imgs)

                # compute loss
                gen_loss = loss_fn(y_true=tf.ones_like(fake_preds),
                                   y_pred=fake_preds)

            # compute gen gradients
            gen_gradients = gen_tape.gradient(gen_loss,
                                              net_g.trainable_variables)

            # update gen weights
            gen_optimizer.apply_gradients(zip(gen_gradients,
                                              net_g.trainable_variables))

            # update gen metrics
            gen_loss_tracker.update_state(gen_loss)

        # generate and save sample images per epoch
        test_generated_imgs = net_g(latent_code4visualization)
        test_generated_imgs = (((test_generated_imgs+1.)/2.) * 255.).numpy()
        plt.figure(figsize=(5, 5))
        for i in range(test_generated_imgs.shape[0]):
            plt.subplot(5, 5, i+1)
            plt.imshow(test_generated_imgs[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.savefig(cfg['img_save_dir']+'/'+cfg['experiment_name'])
        plt.close()
        
        # display and record metrics at the end of each epoch.
        with summary_writer.as_default():
            tf.summary.scalar('disc_loss', disc_loss_tracker.result(), 
                              step=epoch)
            tf.summary.scalar('gen_loss', gen_loss_tracker.result(), 
                              step=epoch)
            tf.summary.image(name='test_samples', data=test_generated_imgs,
                             max_outputs=test_generated_imgs.shape[0], 
                             step=epoch)

        disc_loss = disc_loss_tracker.result()
        gen_loss = gen_loss_tracker.result()
        print(f'epoch: {epoch}, disc_loss: {disc_loss:.4f}, gen_loss: {gen_loss:.4f}')

        # reset metric states
        disc_loss_tracker.reset_state()
        gen_loss_tracker.reset_state()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    with open(_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_ds = load_data(cfg)

    train(cfg, train_ds)