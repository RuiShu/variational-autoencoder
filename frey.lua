require 'pl'
require 'optim'
require 'utils.FreyManager'
require 'utils.Logger'
path.mkdir('save')
local c = require 'trepl.colorize'

local cmd = lapp[[
--gpu          (default 1)     | 1 if gpu, 0 if not gpu
--model        (default Vae)   | which model to use
--hSize        (default 400)   | size of hidden layer
--zSize        (default 2)     | size of latent layer
--mSize        (default 10)    | number of mixtures
--learningRate (default 0.001) | learning rate
--maxEpoch     (default 2000)  | number of total epochs
--epochStep    (default 500)   | number of steps before each step decay
--epochDecay   (default 0.1)   | epoch step decay rate
--saveStep     (default 50)    | number of steps before each save
--validStep    (default -1)    | number of steps before validation
--showVis                      | show training visualization
--height       (default 28)    | height of input image
--width        (default 20)    | width of input image
]]
local struct = {x = 560, h = cmd.hSize, z = cmd.zSize, m = cmd.mSize}
require('models.'..cmd.model)
local data = FreyManager(200)
local vae = _G[cmd.model](struct)
local logger = Logger(cmd)
nn.initialize(vae.model, 'msr')

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
   -- training
   epoch = epoch + 1
   data:shuffle()
   while data:inEpoch() do
      local minibatch = data:next()
      local feval = function(x) return vae:feval(x, minibatch) end
      optim.adam(feval, vae.parameters, config, state)
      logger:receiveRecord(vae:sendRecord())
   end
   logger:log()
   -- validation
   if epoch % cmd.validStep == 0 and cmd.validStep > 0 then
      print(c.green "Validating")
      data:shuffleValid(200)
      local kldErr, bceErr, nElbo = 0, 0, 0
      while data:inEpoch() do
         local minibatch = data:next()
         local e1, e2, e3 = vae:loss(minibatch)
         kldErr = kldErr + e1
         bceErr = bceErr + e2
         nElbo = nElbo + e3
      end
      print(c.red '==> '..'KLD: '..kldErr/10000)
      print(c.red '==> '..'BCE: '..bceErr/10000)
      print(c.red '==> '..'nElbo: '..nElbo/10000)
   end
   -- adjust learning rate and saving
   if epoch % cmd.epochStep == 0 then
      config.learningRate = config.learningRate * cmd.epochDecay
      print(c.green "New learning rate:"..config.learningRate)
   end
   if epoch % cmd.saveStep == 0 then
      local file = 'save/'..cmd.model..'_z'..cmd.zSize
      if string.find(cmd.model, 'Gmm') then file = file..'_m'..cmd.mSize end
      file = file..'.t7'
      torch.save(file, {config=config, state=state, vae=vae})
      print(c.green "Saved checkpoint to "..file)
   end
end
