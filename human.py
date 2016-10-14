import gym

def get_action(string):
    while True:
        if string == 'w':
            return 3
        if string == 'd':
            return 2
        if string == 's':
            return 1
        if string == 'a':
            return 0
        else:
            print 'Invalid command {}:\nw - up\ns - down\nd - right\na - left'.format(string)
            string = raw_input('Give command: ')
        

def run():
    
    env = gym.make('FrozenLake-v0')

    try:
        command = ''
        while command != 'exit':
            observation = env.reset()
            env.render(mode='human')
            command = raw_input()

            r_sum = 0
            
            while command != 'exit':
                action = get_action(command)
                observation, reward, done, info = env.step(action)
                env.render(mode='human')

                r_sum += reward
                if done:
                    if r_sum == 1:
                        print 'You Win!'
                    else:
                        print 'You Lose!'
                    break

                command = raw_input()
            
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    run()
