local Vae = torch.class("Vae")
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function Vae:__init(struct)
   -- build model
   self.encoder, self.decoder, self.model = self:build(struct)
   self.kld = nn.KLDCriterion()
   self.bce = nn.BCECriterion()
   self.kldWeight = 1
   self.bceWeight = 1
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
   local kldErr = self.kld:forward(mulv)
   local bceErr = self.bce:forward(recon, input)
   -- backward
   local dmulv = self.kld:backward(mulv, pmulv)
   local drecon = self.bce:backward(recon, input)
   error_grads = {dmulv:mul(self.kldWeight), drecon:mul(self.bceWeight)}
   self.model:backward(input, error_grads)
   -- record
   local nElbo = kldErr + bceErr
   self:record(kldErr, bceErr, nElbo)
   return nelbo, self.gradients
end

function Vae:loss(minibatch)
   local input = minibatch[1]
   -- forward
   local mulv, recon = unpack(self.model:forward(input))
   local pmulv = mulv:clone():zero()
   local kldErr = self.kld:forward(mulv, pmulv)
   local bceErr = self.bce:forward(recon, input)
   local nElbo = kldErr + bceErr
   return kldErr, bceErr, nElbo
end

function Vae:record(kldErr, bceErr, nElbo)
   -- record
   self.bceErr = bceErr
   self.kldErr = kldErr
   self.nElbo = nElbo
end

function Vae:sendRecord()
   local comm = {}
   comm.bceErr = self.bceErr
   comm.kldErr = self.kldErr
   comm.nElbo = self.nElbo
   comm.decoder = self.decoder
   return comm
end

function Vae:cuda()
   require 'cunn'
   self.model:cuda()
   self.bce:cuda()
   self.kld:cuda()
   self.parameters, self.gradients = self.model:getParameters()
   return self
end
