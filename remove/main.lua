-- Originally by Joost van Amersfoort - <joost@joo.st>
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
nngraph.setDebug(false)

local cmd = torch.CmdLine()
cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'gpu indicator')
cmd:option('-log', 0, 'log indicator')
params = cmd:parse(arg)
params_string = cmd:string('experiment', params, {log=true})

if params.log > 0 then
   paths.mkdir('save')
   cmd:log('save/' .. params_string .. '.log', params)
end
 
--For loading data files
require 'load'
-- Load VAE relevant code
local VAE = require 'VAE'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'

local continuous = false
data = load(continuous)

local input_size = data.train:size(2)
local latent_variable_size = 2
local hidden_layer_size = 400

local batch_size = 200

-- torch.manualSeed(1)

local encoder = VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
local decoder = VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, continuous)

-- end-to-end model
local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})
local reconstruction = decoder(z)
local model = nn.gModule({input},{reconstruction, z, mean, log_var})
criterion = nn.BCECriterion()
criterion.sizeAverage = false

KLD = nn.KLDCriterion()
zeros = torch.Tensor(200, 2):fill(0)
-- enable GPU 
if params.gpu > 0 then
   require 'cutorch'
   require 'cunn'
   model:cuda()
   criterion:cuda()
   KLD:cuda()
   data.train = data.train:cuda()
   zeros = zeros:cuda()
end

local parameters, gradients = model:getParameters()

local config = {
   learningRate = 0.001
}

local state = {}
local epoch = 0



while true do
   epoch = epoch + 1
   local nlowerbound = 0
   local tic = torch.tic()

   local shuffle = torch.randperm(data.train:size(1))

   -- This batch creation is inspired by szagoruyko CIFAR example.
   local indices = torch.randperm(data.train:size(1)):long():split(batch_size)
   indices[#indices] = nil
   local N = #indices * batch_size

   local tic = torch.tic()
   for t,v in ipairs(indices) do
      xlua.progress(t, #indices)

      local inputs = data.train:index(1,v)

      local opfunc = function(x)
	 if x ~= parameters then
	    parameters:copy(x)
	 end

	 model:zeroGradParameters()
	 local reconstruction, z, mean, log_var
	 reconstruction, z, mean, log_var = unpack(model:forward(inputs))

	 local err = criterion:forward(reconstruction, inputs)
	 local df_dw = criterion:backward(reconstruction, inputs)
	 
	 local KLDerr = KLD:forward(mean, log_var)
	 local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))

	 
	 error_grads = {df_dw, zeros, dKLD_dmu, dKLD_dlog_var}

	 model:backward(inputs, error_grads)
	 local batchnlowerbound = err + KLDerr
	 return batchnlowerbound, gradients
      end

      x, batchnlowerbound = optim.adam(opfunc, parameters, config, state)
      nlowerbound = nlowerbound + batchnlowerbound[1]
   end

   print("Epoch: " .. epoch .. " lowerbound: " .. -nlowerbound/N .. " time: " .. torch.toc(tic)) 

   if nlowerboundlist then
      nlowerboundlist = torch.cat(nlowerboundlist,torch.Tensor(1,1):fill(nlowerbound/N),1)
   else
      nlowerboundlist = torch.Tensor(1,1):fill(nlowerbound/N)
   end

   if epoch % 2 == 0 then
      -- torch.save('save/parameters.t7', parameters)
      -- torch.save('save/state.t7', state)
      -- torch.save('save/nlowerbound.t7', torch.Tensor(nlowerboundlist))
      torch.save('save/VAE_z'.. latent_variable_size .. '.t7', 
		 {state=state,
		  nlowerbound=nlowerbound,
		  model=model,
		  encoder=encoder,
		  decoder=decoder})
   end
end
