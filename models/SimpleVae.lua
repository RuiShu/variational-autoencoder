require 'models.Vae'
local SimpleVae, parent = torch.class('SimpleVae', 'Vae')
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function SimpleVae:__init(struct)
   -- build model
   self.encoder, self.decoder, self.model = self:build(struct)
   self.kld = nn.SimpleKLDCriterion()
   self.bce = nn.BCECriterion()
   self.bce.sizeAverage = false
   self.parameters, self.gradients = self.model:getParameters()
end

function SimpleVae:build(struct)
   -- construct self.encoder
   local encoder = nn.Sequential()
   encoder:add(nn.Linear(struct.x, struct.h))
   encoder:add(nn.ReLU(true))
   local mean_logvar = nn.ConcatTable()
   mean_logvar:add(nn.Linear(struct.h, struct.z))
   mean_logvar:add(nn.Linear(struct.h, struct.z))
   encoder:add(mean_logvar)
   -- construct self.decoder
   local decoder = nn.Sequential()
   decoder:add(nn.Linear(struct.z, struct.h))
   decoder:add(nn.ReLU(true))
   decoder:add(nn.Linear(struct.h, struct.x))
   decoder:add(nn.Sigmoid(true))
   -- combine the two
   local input = nn.Identity()()
   local mu, logv = encoder(input):split(2)
   local code = nn.SimpleSampler()({mu, logv})
   local recon = decoder(code)
   local model = nn.gModule({input},{mu, logv, recon})
   return encoder, decoder, model
end

function SimpleVae:feval(x, minibatch)
   local input = minibatch[1]
   if self.parameters ~= x then
      self.parameters:copy(x)
   end
   self.model:zeroGradParameters()
   -- forward
   local mu, logv, recon = unpack(self.model:forward(input))
   local kld_err = self.kld:forward(mu, logv)
   local bce_err = self.bce:forward(recon, input)
   -- backward
   local dmu, dlogv = unpack(self.kld:backward(mu, logv))
   local drecon = self.bce:backward(recon, input)
   error_grads = {dmu, dlogv, drecon}
   self.model:backward(input, error_grads)
   -- record
   local nelbo = kld_err + bce_err
   self:record(kld_err, bce_err, nelbo)
   return nelbo, self.gradients
end

