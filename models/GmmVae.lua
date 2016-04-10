local GmmVae = torch.class("GmmVae")
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function GmmVae:__init(struct)
   -- build model
   self.prior, self.encoder, self.decoder, self.model = self:build(struct)
   self.kld = nn.KLDCriterion()
   self.bce = nn.BCECriterion()
   self.bce.sizeAverage = false
   self.parameters, self.gradients = self.model:getParameters()
end

function GmmVae:build(struct)
   -- construct self.encoder
   local encoder = nn.Sequential()
   encoder:add(nn.Linear(struct.x, struct.h))
   encoder:add(nn.ReLU(true))
   encoder:add(nn.Linear(struct.h, struct.z*2))
   encoder:add(nn.View(2, struct.z))
   -- construct self.decoder
   local decoder = nn.Sequential()
   decoder:add(nn.Linear(struct.z, struct.h))
   decoder:add(nn.ReLU(true))
   decoder:add(nn.Linear(struct.h, struct.x))
   decoder:add(nn.Sigmoid(true))
   -- construct self.prior
   local prior = nn.Sequential()
   prior:add(nn.Linear(struct.x, struct.h))
   prior:add(nn.ReLU(true))
   prior:add(nn.Linear(struct.h, struct.h))
   prior:add(nn.ReLU(true))
   prior:add(nn.Linear(struct.h, struct.h))
   prior:add(nn.ReLU(true))
   prior:add(nn.Linear(struct.h, 2*struct.m*struct.z))
   prior:add(nn.View(2*struct.m, struct.z))
   -- combine the three
   local mask, input = nn.Identity()(), nn.Identity()()
   local pmulv = prior(mask)
   local mulv = encoder(input)
   local code = nn.Sampler()(mulv)
   local recon = decoder(code)
   local model = nn.gModule({mask, input},{pmulv, mulv, recon})
   return prior, encoder, decoder, model
end

function GmmVae:feval(x, minibatch)
   local input = minibatch[1]
   if self.parameters ~= x then
      self.parameters:copy(x)
   end
   self.model:zeroGradParameters()
   -- forward
   local mask = input:clone():zero()
   local pmulv, mulv, recon = unpack(self.model:forward({mask, input}))
   local kld_err = self.kld:forward(mulv, pmulv)
   local bce_err = self.bce:forward(recon, input)
   -- backward
   local dmulv, dpmulv = self.kld:backward(mulv, pmulv)
   local drecon = self.bce:backward(recon, input)
   error_grads = {dpmulv, dmulv, drecon}
   self.model:backward(input, error_grads)
   -- record
   local nelbo = kld_err + bce_err
   self:record(kld_err, bce_err, nelbo)
   return nelbo, self.gradients
end

function GmmVae:record(bceErr, kldErr, nElbo)
   -- record
   self.bceErr = bceErr
   self.kldErr = kldErr
   self.nElbo = nElbo
end

function GmmVae:sendRecord()
   local comm = {}
   comm.bceErr = self.bceErr
   comm.kldErr = self.kldErr
   comm.nElbo = self.nElbo
   comm.decoder = self.decoder
   return comm
end

function GmmVae:cuda()
   require 'cunn'
   self.model:cuda()
   self.bce:cuda()
   self.kld:cuda()
   self.parameters, self.gradients = self.model:getParameters()
   return self
end
