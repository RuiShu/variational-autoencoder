local Vae = torch.class("Vae")
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
   -- construct self.encoder
   local encoder = nn.Sequential()
   encoder:add(nn.Linear(struct.x, struct.h)):add(nn.ReLU(true))
   encoder:add(nn.Linear(struct.h, 2*struct.z))
   encoder:add(nn.View(2, struct.z))
   -- construct self.decoder
   local decoder = nn.Sequential()
   decoder:add(nn.Linear(struct.z, struct.h)):add(nn.ReLU(true))
   decoder:add(nn.Linear(struct.h, struct.x))
   decoder:add(nn.Sigmoid(true))
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
   self.parameters, self.gradients = self.model:getParameters()
   return self
end
