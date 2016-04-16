require 'models.CnnVae'
local CnnVaeFrey, parent = torch.class('CnnVaeFrey', 'CnnVae')
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function CnnVaeFrey:__init(struct)
   -- build model
   self.encoder, self.decoder, self.model = self:build(struct)
   self.kld = nn.KLDCriterion()
   self.rec = nn.GaussianCriterion()
   self.kldWeight = 1
   self.recWeight = 10
   self.rec.sizeAverage = false
   self.parameters, self.gradients = self.model:getParameters()
end

function CnnVaeFrey:build(struct)
   local encoder = nn.Sequential()
   -- conv
   encoder:add(nn.View(1,28,20))
      :add(nn.SpatialConvolution(  1, 50, 2,2, 2,2)):add(nn.ReLU(true)) -- 14x10
      :add(nn.SpatialConvolution( 50,100, 2,2, 2,2)):add(nn.ReLU(true)) -- 7 x 5
      :add(nn.SpatialConvolution(100,200, 3,3, 2,2)):add(nn.ReLU(true)) -- 3 x 2
      :add(nn.SpatialConvolution(200,400, 2,3, 2,2)):add(nn.ReLU(true)) -- 1 x 1
   -- linear
      :add(nn.View(400))
      :add(nn.Linear(400,2*struct.z))
      :add(nn.View(2,struct.z))
   local decoder = nn.Sequential()
   -- linear
      :add(nn.Linear(struct.z, struct.h)):add(nn.ReLU(true))
      :add(nn.View(400,1,1))
   -- conv
      :add(nn.SpatialFullConvolution(400,200, 2,3, 2,2)):add(nn.ReLU(true))
      :add(nn.SpatialFullConvolution(200,100, 3,3, 2,2)):add(nn.ReLU(true))
      :add(nn.SpatialFullConvolution(100, 50, 2,2, 2,2)):add(nn.ReLU(true))
      :add(nn.SpatialFullConvolution( 50,  2, 2,2, 2,2))
      :add(nn.View(2,1,28,20))
      :add(nn.SplitTable(1,4))
      :add(nn.ParallelTable()
              :add(nn.Contiguous())
              :add(nn.Identity()))
      :add(nn.ParallelTable()
              :add(nn.Tanh(true))
              :add(nn.Clamp(-5, 100)))
      :add(nn.JoinTable(1, 3))
      :add(nn.View(2, 560))
   -- combine the two
   local input = nn.Identity()()
   local mulv = encoder(input)
   local code = nn.Sampler()(mulv)
   local recon = decoder(code)
   local model = nn.gModule({input},{mulv, recon})
   return encoder, decoder, model
end
