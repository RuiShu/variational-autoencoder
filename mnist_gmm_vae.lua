require 'utils.MnistManager'
require 'utils.Logger'
require 'models.GmmVae'
require 'pl'
require 'optim'
require 'lfs'
lfs.mkdir('save')
local c = require 'trepl.colorize'

cmd = lapp[[
--gpu           (default 1)     | 1 if gpu, 0 if not gpu
--h_size        (default 400)   | size of hidden layer
--z_size        (default 2)     | size of latent layer
--m_size        (default 10)    | number of mixtures
--learning_rate (default 0.001) | learning rate
--max_epoch     (default 400)   | number of total epochs
--epoch_step    (default 100)   | number of steps before each step decay
--epoch_decay   (default 0.1)   | epoch step decay rate
--save_step     (default 50)    | number of steps before each save
]]

local data = MnistManager(200)
local vae = GmmVae({x=784, h=cmd.h_size, z=cmd.z_size, m=cmd.m_size})
local logger = Logger(cmd)
if cmd.gpu == 1 then
   require 'cunn'
   data:cuda()
   vae:cuda()
   logger:cuda()
end

local config = {learningRate = cmd.learning_rate}
local state = {}

local epoch = 0
while epoch < cmd.max_epoch do
   epoch = epoch + 1
   data:shuffle()
   while data:inEpoch() do
      local minibatch = data:next()
      local feval = function(x) return vae:feval(x, minibatch) end
      optim.adam(feval, vae.parameters, config, state)
      logger:receiveRecord(vae:sendRecord())
   end
   logger:log()
   if epoch % cmd.epoch_step == 0 then
      config.learningRate = config.learningRate * cmd.epoch_decay
      print(c.green "New learning rate:"..config.learningRate)
   end
   if epoch % cmd.save_step == 0 then
      torch.save('save/GmmVae.t7',
                 {config=config,
                  state=state,
                  vae=vae})
      print(c.green "Saved checkpoint")
   end
end
