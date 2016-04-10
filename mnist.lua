require 'lfs'
require 'pl'
require 'optim'
require 'utils.MnistManager'
require 'utils.Logger'
lfs.mkdir('save')
local c = require 'trepl.colorize'

local cmd = lapp[[
--gpu          (default 1)     | 1 if gpu, 0 if not gpu
--model        (default Vae)   | which model to use
--hSize        (default 400)   | size of hidden layer
--zSize        (default 2)     | size of latent layer
--mSize        (default 10)    | number of mixtures
--learningRate (default 0.001) | learning rate
--maxEpoch     (default 400)   | number of total epochs
--epochStep    (default 100)   | number of steps before each step decay
--epochDecay   (default 0.1)   | epoch step decay rate
--saveStep     (default 50)    | number of steps before each save
--showVis      (default true)  | show training visualization
]]
local struct = {x = 784, h = cmd.hSize, z = cmd.zSize, m = cmd.mSize}
require('models.'..cmd.model)
local data = MnistManager(200)
local vae = _G[cmd.model](struct)
local logger = Logger(cmd)

if cmd.gpu == 1 then
   require 'cunn'
   data:cuda()
   vae:cuda()
   logger:cuda()
end

local config = {learningRate = cmd.learningRate}
local state = {}

local epoch = 0
while epoch < cmd.maxEpoch do
   epoch = epoch + 1
   data:shuffle()
   while data:inEpoch() do
      local minibatch = data:next()
      local feval = function(x) return vae:feval(x, minibatch) end
      optim.adam(feval, vae.parameters, config, state)
      logger:receiveRecord(vae:sendRecord())
   end
   logger:log()
   if epoch % cmd.epochStep == 0 then
      config.learningRate = config.learningRate * cmd.epochDecay
      print(c.green "New learning rate:"..config.learningRate)
   end
   if epoch % cmd.saveStep == 0 then
      torch.save('save/'..cmd.model..'.t7',
                 {config=config,
                  state=state,
                  vae=vae})
      print(c.green "Saved checkpoint")
   end
end
