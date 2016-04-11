require 'models.Vae'
local CnnVae, parent = torch.class('CnnVae', 'Vae')
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function CnnVae:__init(struct)
   -- build model
   self.encoder, self.decoder, self.model = self:build(struct)
   self.kld = nn.KLDCriterion()
   self.bce = nn.BCECriterion()
   self.bce.sizeAverage = false
   self.parameters, self.gradients = self.model:getParameters()
end

function CnnVae:build(struct)
   local encoder = nn.Sequential()
   -- conv
   encoder:add(nn.View(1,28,28))
   encoder:add(nn.SpatialConvolution(  1, 50, 2,2, 2,2)):add(nn.ReLU(true))
   encoder:add(nn.SpatialConvolution( 50,100, 2,2, 2,2)):add(nn.ReLU(true))
   encoder:add(nn.SpatialConvolution(100,200, 3,3, 2,2)):add(nn.ReLU(true))
   encoder:add(nn.SpatialConvolution(200,400, 3,3, 2,2)):add(nn.ReLU(true))
   -- linear
   encoder:add(nn.View(400))
   encoder:add(nn.Linear(400,2*struct.z))
   encoder:add(nn.View(2,struct.z))
   local decoder = nn.Sequential()
   -- linear
   decoder:add(nn.Linear(struct.z, struct.h)):add(nn.ReLU(true))
   decoder:add(nn.View(400,1,1))
   -- conv
   decoder:add(nn.SpatialFullConvolution(400,200, 3,3, 2,2)):add(nn.ReLU(true))
   decoder:add(nn.SpatialFullConvolution(200,100, 3,3, 2,2)):add(nn.ReLU(true))
   decoder:add(nn.SpatialFullConvolution(100, 50, 2,2, 2,2)):add(nn.ReLU(true))
   decoder:add(nn.SpatialFullConvolution( 50,  1, 2,2, 2,2))
   decoder:add(nn.View(784))
   decoder:add(nn.Sigmoid())
   -- combine the two
   local input = nn.Identity()()
   local mulv = encoder(input)
   local code = nn.Sampler()(mulv)
   local recon = decoder(code)
   local model = nn.gModule({input},{mulv, recon})
   return encoder, decoder, model
end

function CnnVae:feval(x, minibatch)
   local input = minibatch[1]
   if self.parameters ~= x then
      self.parameters:copy(x)
   end
   self.model:zeroGradParameters()
   -- forward
   local mulv, recon = unpack(self.model:forward(input))
   local pmulv = mulv:clone():zero()
   local kld_err = self.kld:forward(mulv, pmulv)
   local bce_err = self.bce:forward(recon, input)
   -- backward
   local dmulv = self.kld:backward(mulv, pmulv)
   local drecon = self.bce:backward(recon, input)
   error_grads = {dmulv, drecon}
   self.model:backward(input, error_grads)
   -- record
   local nelbo = kld_err + bce_err
   self:record(kld_err, bce_err, nelbo)
   return nelbo, self.gradients
end

function CnnVae:cuda()
   parent.cuda(self)
   require 'cudnn'
   cudnn.benchmark = true
   cudnn.fastest = true
   cudnn.convert(self.encoder, cudnn)
   cudnn.convert(self.decoder, cudnn)
   return self
end
