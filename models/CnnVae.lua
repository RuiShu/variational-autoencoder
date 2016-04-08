local Vae = torch.class("CnnVae")
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function Vae:__init(struct)
   -- build model
   self.encoder, self.decoder, self.model = self:build(struct)
   self.kld = nn.KLDCriterion()
   self.bce = nn.BCECriterion()
   self.bce.sizeAverage = false
   self.parameters, self.gradients = self.model:getParameters()
end

function Vae:build(struct)
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

function Vae:feval(x, minibatch)
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

function Vae:record(bce_err, kld_err, nelbo)
   -- record
   self.bce_status = self.bce_status or bce_err
   self.kld_status = self.kld_status or kld_err
   self.elbo_status = self.elbo_status or -nelbo
   self.kld_status = 0.99*self.kld_status + 0.01*kld_err
   self.bce_status = 0.99*self.bce_status + 0.01*bce_err
   self.elbo_status = 0.99*self.elbo_status - 0.01*nelbo
end

function Vae:log()
   self.epoch = self.epoch or 0
   self.epoch = self.epoch + 1
   print(c.green 'Epoch: '..self.epoch)
   print(c.red '==> '..'Elbo: '..self.elbo_status/200)
   print(c.red '==> '..'KLD: '..self.kld_status/200)
   print(c.red '==> '..'BCE: '..self.bce_status/200)
end

function Vae:cuda()
   require 'cunn'
   self.model:cuda()
   self.bce:cuda()
   self.kld:cuda()
   require 'cudnn'
   -- cudnn.benchmark = true
   cudnn.fastest = true
   cudnn.convert(self.encoder, cudnn)
   cudnn.convert(self.decoder, cudnn)
   self.parameters, self.gradients = self.model:getParameters()
   return self
end
