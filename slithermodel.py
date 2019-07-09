import gym
import universe
import random
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.utils import plot_model, to_categorical
import pickle

# reinforcement learning step
def get_obs_len_reward(observation_n, reward_n, done_n, total_reward_sum, total_length):
    '''Proposed reward model
    If increase in length >= 25: Reward = +3
    If increase in length >= 10 and < 25: Reward = +2
    If increase in length > 0 and < 10: Reward = +1
    If increase in length < 0 (decrease): Reward = -3
    If killed, Reward = -10 * total_length
    '''
    total_length += reward_n
    step_reward = 0
    if reward_n >= 25:
        step_reward = 3
    elif reward_n >= 10 and reward_n < 25:
        step_reward = 2
    elif reward_n > 0 and reward_n < 10:
        step_reward = 1
    elif reward_n < 0:
        step_reward = -3
    if done_n == True:
        step_reward = -100
    
    total_reward_sum += step_reward

    return step_reward, total_reward_sum, total_length

def action_to_reward_model(input_shape):
    '''Map action to reward using a multilayer perceptron'''
    model = Sequential()
    model.add(Dense(128, input_dim = input_shape, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(1, activation="linear"))
    with open('mlpmodel.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model

def image_to_action_model(input_shape, filters=(16, 32, 64)):
    '''Map image to reward using a Deep CNN'''
    inputs = Input(shape=input_shape)
    for (index, filter) in enumerate(filters):
        if index == 0:
            x = inputs
        x = Conv2D(filter, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis = -1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis = -1)(x)
    x = Dropout(0.2)(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis = -1)(x)
    x = Dropout(0.2)(x)
    x = Dense(4)(x)
    x = Activation('relu')(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs, x)
    
    with open('cnnmodel.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model

def build_model(action_input_shape, image_input_shape, output_shape):
    '''Create the hybrid model'''
    action_to_reward_X = action_to_reward_model(action_input_shape)
    image_to_reward_X = image_to_action_model(image_input_shape)
    combined_input = concatenate([action_to_reward_X.output, image_to_reward_X.output])
    x = Dense(64, activation="relu")(combined_input)
    x = Dense(16, activation="relu")(x)
    x = Dense(4, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=[action_to_reward_X.input, image_to_reward_X.input], outputs=x)
    return model

def train_model(model, observation_list, step_reward_list, step_action_list):
    image_X = np.asarray(observation_list)
    image_X = image_X[:, :, :, np.newaxis]
    action_x = np.asarray(step_action_list)
    reward_y = np.asarray(step_reward_list)
    if model == None:
        model = build_model(action_input_shape= 3, image_input_shape=(300, 502, 1), output_shape=1)
        opt = Adam(lr=0.0001, decay=1e-3 / 200)
        model.compile(loss="mean_squared_error", optimizer=opt)
    with open('mixedmodel.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    hist = model.fit([action_x, image_X], reward_y, epochs=20, batch_size=32)
    return model, hist

def play(model, run_num, phase):
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()
    n = 0
    total_reward_sum = 0
    reward_n = [0]
    done_n = [False]
    step_reward = 0
    total_length = 10
    available_actions = [[231, 196, 0], [231, 196, 1], [231, 236, 0], [231, 236, 1], [231, 276, 0], [231, 276, 1],
                         [271, 196, 0], [271, 196, 1], [271, 236, 0], [271, 236, 1], [271, 276, 0], [271, 276, 1],
                         [311, 196, 0], [311, 196, 1], [311, 236, 0], [311, 236, 1], [311, 276, 0], [311, 276, 1]]
    slow_actions = [[231, 196, 0], [231, 236, 0], [231, 276, 0],
                    [271, 196, 0], [271, 236, 0], [271, 276, 0],
                    [311, 196, 0], [311, 236, 0], [311, 276, 0]]
    observation_list = []
    step_reward_list = []
    step_action_list = []
    event = np.asarray([271, 236, 0])


    action_n = [[('PointerEvent', 271, 236, 0)]]
    done_n[0] = False
    while done_n[0] == False: 
        n+=1
        if (n>1):
            if phase == 'train' or len(observation_list) == 0:
                event = random.choice(available_actions)
                action_n = [[('PointerEvent', event[0], event[1], event[2])]]
                event = np.asarray(event)

            elif phase == 'test':
                obs = np.asarray(observation_list[-1])
                obs = obs[np.newaxis, :, :, np.newaxis]
                for act in slow_actions:
                    max_reward = 0
                    event = []
                    reward_action = model.predict([[np.asarray(act)], obs])[0][0]
                    if (reward_action > max_reward):
                        max_reward = reward_action
                        action_n = [[('PointerEvent', act[0], act[1], act[2])]]
                        event = np.asarray(act)

                if len(event) == 0:
                    event = random.choice(slow_actions)
                    action_n = [[('PointerEvent', event[0], event[1], event[2])]]
                    event = np.asarray(event)

        if (observation_n[0] != None):
            step_reward, total_reward_sum, total_length = get_obs_len_reward(observation_n, reward_n[0], done_n[0], total_reward_sum, total_length)

        # save the new variables for each iteration
        observation_n, reward_n, done_n, info = env.step(action_n)
        if done_n[0] == True and len(step_reward_list) > 0:
            step_reward_list[-1] = -100
        # for purposes of visualization only
        if (observation_n[0] != None and step_reward != 0 ):
            arr = np.asarray(observation_n[0]['vision'])
            arr = arr[86:386, 20:522, :]
            R = arr[: , : , 0] * 0.299
            G = arr[: , : , 1] * 0.587
            B = arr[: , : , 2] * 0.114
            gray_imgarr = (R + G + B)
            observation_list.append(gray_imgarr)
            step_action_list.append(event)
            step_reward_list.append(step_reward)
            arr2im = Image.fromarray(arr)
            arr2im.save('observation_n.jpg')
        env.render()

    return observation_list, step_reward_list, step_action_list, total_length


if __name__ == '__main__':
    # Number of times to play the game
    num_play = 1000
    total_reward = []
    total_length_list = []
    loss_history = []
    try:
        with open('obsFile', 'rb') as lf:
            observation_list = pickle.load(lf)
    except:
        observation_list = []

    try:
        with open('rewardlistFile', 'rb') as lf:
            step_reward_list = pickle.load(lf)
        with open('actionlistFile', 'rb') as lf:
            step_action_list = pickle.load(lf)
    except:
        step_reward_list = []
        step_action_list = []

    model = None
    for x in range(1, num_play):
        if x <= 1:
            print('-'*100, 'training', x)
            observation_list_train, step_reward_list_train, step_action_list_train, total_length = play(None, x, 'train')
            observation_list.extend(observation_list_train)
            with open('obsFile', 'wb') as lf:
                pickle.dump(observation_list, lf)
            step_reward_list.extend(step_reward_list_train)
            with open('rewardlistFile', 'wb') as lf:
                pickle.dump(step_reward_list, lf)
            step_action_list.extend(step_action_list_train)
            with open('actionlistFile', 'wb') as lf:
                pickle.dump(step_action_list, lf)

            print('==============> step_reward_list', step_reward_list_train)
            print('==============> step_action_list', step_action_list_train)
            print('==============> total_length', total_length)


        if (len(observation_list) > 0 and x > 1 and x % 5 == 0):
            try:
                model = load_model('slither_model.h5')
            except:
                model = None

            model, hist = train_model(model, observation_list, step_reward_list, step_action_list)
            # Save the model
            model.save('slither_model.h5')
            loss_history.append(hist.history['loss'][-1])
            with open('historyFile', 'wb') as hf:
                pickle.dump(loss_history, hf)

        if model != None and x > 1:
            print('-'*100, 'testing')
            observation_list_x, step_reward_list_x, step_action_list_x, total_length = play(model, x, 'test')
            observation_list.extend(observation_list_x)
            step_reward_list.extend(step_reward_list_x)
            step_action_list.extend(step_action_list_x)
            print('==============>  step_action_list_x', step_action_list_x)
            print('==============> step_reward_list', step_reward_list_x)
            print('==============> total_length', total_length)
            total_reward = np.append(total_reward, np.sum(np.asarray(step_reward_list_x)))
            total_length_list = np.append(total_length_list, total_length)
            with open('rewardsFile', 'wb') as rf:
                pickle.dump(total_reward, rf)
            with open('lengthFile', 'wb') as lf:
                pickle.dump(total_length_list, lf)
            print('------- iteration', x)
            print('------- iteration reward', np.sum(np.asarray(step_reward_list_x)))
            print('------- iteration length', total_length)
            print('------- total_reward', total_reward)
            print('------- total_length_list', total_length_list)
        
        with open('obsFile', 'wb') as lf:
            pickle.dump(observation_list, lf)
        with open('rewardlistFile', 'wb') as lf:
            pickle.dump(step_reward_list, lf)
        with open('actionlistFile', 'wb') as lf:
            pickle.dump(step_action_list, lf)