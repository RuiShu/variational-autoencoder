local Vae = torch.class("Vae")
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function Vae:__init(struct)
   -- build model
   self.encoder, self.decoder, self.model = self:build(struct)
   self.kld = nn.KLDCriterion()
   self.rec = nn.BCECriterion()
   self.kldWeight = 1
   self.recWeight = 1
   self.rec.sizeAverage = false
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
   local recErr = self.rec:forward(recon, input)
   -- backward
   local dmulv = self.kld:backward(mulv, pmulv)
   local drecon = self.rec:backward(recon, input)
   errorGrads = {dmulv:mul(self.kldWeight), drecon:mul(self.recWeight)}
   self.model:backward(input, errorGrads)
   -- record
   local nElbo = kldErr + recErr
   self:record(kldErr, recErr, nElbo)
   return nelbo, self.gradients
end

function Vae:loss(minibatch)
   local input = minibatch[1]
   -- forward
   local mulv, recon = unpack(self.model:forward(input))
   local pmulv = mulv:clone():zero()
   local kldErr = self.kld:forward(mulv, pmulv)
   local recErr = self.rec:forward(recon, input)
   local nElbo = kldErr + recErr
   return kldErr, recErr, nElbo
end

function Vae:record(kldErr, recErr, nElbo)
   -- record
   self.recErr = recErr
   self.kldErr = kldErr
   self.nElbo = nElbo
end

function Vae:sendRecord()
   local comm = {}
   comm.recErr = self.recErr
   comm.kldErr = self.kldErr
   comm.nElbo = self.nElbo
   comm.decoder = self.decoder
   return comm
end

function Vae:cuda()
   require 'cunn'
   self.model:cuda()
   self.rec:cuda()
   self.kld:cuda()
   self.parameters, self.gradients = self.model:getParameters()
   return self
end
