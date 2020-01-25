from numpy import *
from matplotlib.pyplot import *
import time


class FluidFlow2D:
    """ 
    This code computes 2D incompressible fluid flow using the 
    Stable Fluids Method outlined here:
    http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/ns.pdf
    
    The code was written in 2016 for ASTRO/EARTH 119
    by Morgan MacLeod, Greg Laughlin
   
   
    """
    
    def __init__(self,ncells=100,xmin=0.,xmax=100.,solver_tolerence=3.e-4):
        """
        Initialization method for the 2D fluid flow class. 
        The grid and solver methods currently assume a square grid
        with dimensions xmin, xmax in the x,y directions
        the grid is made up of ncells total cells in each direction. 
        Individual grid cells are, therefore, square. 
  
        ncells = number of cells in each direction in the 2D grid
        xmin   = min coordinate value in x, y directions
        max    = max coordinate value in the x,y directions
        solver_tolerence = convergence tolerence on the "projection" step 
                           which ensures that the incompressible flow is
                           divergence-free
        """
        self.nx = ncells
        self.ny = ncells
        self.xmin = xmin
        self.xmax = xmax
        self.x = linspace(self.xmin, self.xmax, self.nx)
        self.y = linspace(self.xmin, self.xmax, self.nx)
        self.dx = self.x[1]-self.x[0]
        self.dy = self.dx
        
        self.yy, self.xx = meshgrid(self.x, self.y)
        
        self.tol = solver_tolerence
    
        
    def get_coords(self):
        """ returns the 2D coordinate meshes xx,yy for the grid """
        return self.xx, self.yy
        
    def boundary_condition(self,x,bctype):
        """
        Apply boundary conditinons to a variable x.
        bctype may be either "wrap" or "zero_gradient" 
        """
        # set local variables for use in indexing
        nx = self.nx
        ny = self.ny
        # APPLY TWO DIFFERENT BC OPTIONS
        if(bctype=="wrap"):
            # the x-boundaries
            x[:,0] = x[:,ny-2]
            x[:,ny-1] = x[:,1]
            # the y-boundaries
            x[0,:] = x[nx-2,:]
            x[nx-1,:] = x[1,:]
            
        elif(bctype=="zero_gradient"):
            # the x-boundaries
            x[:,0] = x[:,1]
            x[:,ny-1] = x[:,ny-2]
            # the y-boundaries
            x[0,:] = x[1,:]
            x[nx-1,:] = x[nx-2,:]
        
        else:
            print "BOUNDARY TYPE NOT RECOGNIZED"
        
        return x
    
    
    def setup_initial_condition(self,vxo,vyo,dno,Diff_Coeff=1.e-2,bctype="wrap",dt_initial=1.0):
        """
        Calling this function will set up initial conditions for the fluid flow problem. 
        The parameters are:
        
        vxo  =  2D mesh with the same shape as the grid of x-velocities
        vyo  =  2D mesh of y-velocities
        dno  =  2D mesh of densities
        
        Diff_coeff = viscous diffusion coefficient
        bctype     = type of boundary condition to apply to the flow. Must be a type defined in boundary_condition()
        dt_initial = starting timestep
        """
        # initial and update velocities
        self.vxo = copy(vxo)
        self.vyo = copy(vyo)
        # initial and update densities
        self.dno = copy(dno)
        
        # initialize divergence,q arrays
        self.div = zeros_like(dno)
        self.q   = zeros_like(dno)
        
        #sanitize (make divergence free and apply bcs)
        self.vxo = self.boundary_condition(self.vxo,bctype)
        self.vyo = self.boundary_condition(self.vyo,bctype)
        self.vxo, self.vyo = self.project(self.vxo,self.vyo,self.div,self.q,self.tol)
        
        # define update velocities here
        self.vx  = copy(vxo)
        self.vy  = copy(vyo)
        self.dn  = copy(dno)
        
        #diffusion coefficient
        self.D = Diff_Coeff
        
        # time and timestep
        self.dt = dt_initial
        self.current_time = 0.0
        
    
    #projection step
    def project(self, vx,vy,div,q,tol,nrelax=10000,bctype="wrap"):
        """
        This function solves the elliptic equation
        Div^2 q = Div dot v 
        to ensure that the flow is non-divergent. 
        
        The flow must not be divergent because it is incompressible
        (so it can't converge or diverge). 
        
        The method is iterative: we repeat a calculation until the result (q) is stable. 
        
        The parameters are:
        vx = 2D velocity mesh
        vy = 2D velocity mesh
        div = 2D mesh to be assigned with divergence of velocities (Div dot v)
        q   = 2D mesh to be filled with solution to the elliptic equation
        tol = tolerence that we want the q-array to stabilize to
        nrelax = max number of steps in relaxation iteration
        """
        
        # velocity gradient local grid variables
        grad_qx = zeros_like(q)
        grad_qy = zeros_like(q)
        dx = self.dx
        
        # populate the interior of divergence, then apply bc
        div[1:-1,1:-1] = 1./(2*dx)*(vx[2:,1:-1] - vx[:-2, 1:-1] + vy[1:-1,2:] - vy[1:-1,:-2])
        div = self.boundary_condition(div,"zero_gradient")
    
        #solve Poisson Equation for q using Jacobi Method (iteration)
        qo = copy(q)
        for k in arange(1,nrelax):
            q[1:-1,1:-1] = 0.25*(qo[2:,1:-1] + qo[:-2, 1:-1] + qo[1:-1,2:] + qo[1:-1, :-2] -dx*dx*div[1:-1,1:-1])
            q = self.boundary_condition(q,"zero_gradient")
        
            if(allclose(q,qo,atol=tol)):
                print "... project converged after", k," steps"
                break
            else:
                qo=copy(q)

        #compute gradient of q    
        grad_qx[1:-1,1:-1] = 1/(2*dx) * (q[2:,1:-1]-q[:-2,1:-1])
        grad_qy[1:-1,1:-1] = 1/(2*dx) * (q[1:-1,2:]-q[1:-1,:-2])
        grad_qx = self.boundary_condition(grad_qx,bctype)
        grad_qy = self.boundary_condition(grad_qy,bctype)
            
        #remove gradient field from the velocity arrays
        vx=vx-grad_qx
        vy=vy-grad_qy
        # apply the BC
        vx = self.boundary_condition(vx,bctype)
        vy = self.boundary_condition(vy,bctype)
        return vx,vy
    
    
    def external_acceleration(self,dens):
        """ 
        External acceleration law (depends on density)
        """
        return 0.05*(dens-1.0)
    
    def external_force_update(self,vx,vy,vxo,vyo,dno):
        """ 
        Apply an external force to the flow
        Depends on 2D meshes: vx,vy,vxo,vyo,dno
        """
        vy[1:-1,1:-1] = vyo[1:-1,1:-1] - self.external_acceleration( dno[1:-1,1:-1] )*self.dt
        return vx,vy
    
    def diffusion_FTCS_update(self,X):
        """
        Make a 2D diffusion step to update the velocities 
        Here implimented for a general variable X
        """
        s = self.D*self.dt/self.dx**2
        # apply diffusion equation to quantity X throughout the 
        # "interior" cells
        X[1:-1,1:-1] = X[1:-1,1:-1] \
                   + s*(X[:-2,1:-1] -2*X[1:-1,1:-1] + X[2:,1:-1] +
                       X[1:-1,:-2] - 2*X[1:-1,1:-1] + X[1:-1,2:])
        return X

    def advection_update(self,vx,vy,dn,vxo,vyo,dno):
        """
        Use bilinear interpolation to move fluid velocities 
        in space according to the flow velocity in a cell.
        
        We use the current velocities to estimate the new velocity
        at a given position using the values extrapolated back by time
        dt (distance v*dt). 
        
        Parameters are 2D meshes:
        vx, vy = new x,y velocity
        dn = new density
        vxo, vyo = old x,y velocity
        dno = old density
        """
        dt = self.dt
        nx = self.nx
        ny = self.ny
        x  = self.x
        y  = self.y
        dx = self.dx
        dy = self.dy
        #advection update to the velocities
        for i in arange(1,nx-1):
            for j in arange(1,ny-1):
                x1=x[i]-vxo[i,j]*dt
                y1=y[j]-vyo[i,j]*dt
                ix=i
                if x1>x[i]:
                    ix=i+1        
                jy=j
                if y1>y[j]:
                    jy=j+1           
                t=(x1-x[ix-1])/dx
                u=(y1-y[jy-1])/dy
                vx[i,j]=(1-t)*(1-u)*vxo[ix-1,jy-1]\
                        + t*(1-u)*vxo[ix,jy-1]\
                        + t*u*vxo[ix,jy]\
                        + (1-t)*u*vxo[ix-1,jy]
                vy[i,j]=(1-t)*(1-u)*vyo[ix-1,jy-1]\
                        + t*(1-u)*vyo[ix,jy-1]\
                        + t*u*vyo[ix,jy]\
                        + (1-t)*u*vyo[ix-1,jy]
                dn[i,j]=(1-t)*(1-u)*dno[ix-1,jy-1]\
                        + t*(1-u)*dno[ix,jy-1]\
                        + t*u*dno[ix,jy]\
                        + (1-t)*u*dno[ix-1,jy]
        
        return vx,vy,dn
    
    
    def update_fluid_motion_step(self,flowbc="wrap"):
        """
        Update the fluid motion by one timestep. 
        
        flowbc = choice of boundary condition to be applied to the fluid
        
        Each step follows the following sequence:
        1. apply an external force(if present)
        2. apply diffusion equation to the velocities
        3. apply advection step to move the fluid through the grid
         
        After each sub-step (1-3) above, we apply the boundary condition 
        and apply the projection step to ensure that the updated flow
        distribution remains divergence free. 
        """
    
        # apply any external force
        self.vx,self.vy = self.external_force_update(self.vx,self.vy,self.vxo,self.vyo,self.dno)
        self.vxo = self.boundary_condition(self.vx,flowbc)
        self.vyo = self.boundary_condition(self.vy,flowbc)
    
        #project (ensure non-divergent flow)
        self.vxo, self.vyo = self.project(self.vxo,self.vyo,self.div,self.q,self.tol,bctype=flowbc)
    
        # diffusion update
        self.vx = self.diffusion_FTCS_update(self.vxo)
        self.vy = self.diffusion_FTCS_update(self.vyo)
        self.vxo= self.boundary_condition(self.vx,flowbc)
        self.vyo= self.boundary_condition(self.vy,flowbc)
    
        #project (ensure non-divergent flow)
        self.vxo, self.vyo = self.project(self.vxo,self.vyo,self.div,self.q,self.tol,bctype=flowbc)
       
        # advect the fluid based on its current velocity
        self.vx,self.vy,self.dn = self.advection_update(self.vx,self.vy,self.dn,self.vxo,self.vyo,self.dno)
        self.vxo = self.boundary_condition(self.vx,flowbc)
        self.vyo = self.boundary_condition(self.vy,flowbc)
        self.dno = self.boundary_condition(self.dn,flowbc)
    
        #project (ensure non-divergent flow)
        self.vxo, self.vyo = self.project(self.vxo,self.vyo,self.div,self.q,self.tol,bctype=flowbc)
       
        return None
    
    def choose_new_dt(self, vel_limiter=0.5,acc_limiter=0.25,diff_limiter=0.5):
        """
        Choose a new timestep based on stability criteria:
        
        velocity criterion:     dx/v > dt
        acceleration criterion: sqrt(2dx/acc) > dt
        diffusion criterion:    dx**2 / (4 D) > dt
        
        Each of these finds the "worst-case" value in the 2D grid. 
        
        Parameters are:
        vel_limiter = factor to multiply by the critical velocity timestep. Should be in the range 0-1
        acc_limiter = (same as above but for the acceleration criterion)
        diff_limiter = (same as above for the diffusion criterion)
        """
        
        # local variables for timestep
        dt = self.dt
        dx = self.dx
        
        #next check for timestep update
        vxmax = amax(self.vxo)
        vymax = amax(self.vyo)
        vmax = max(vxmax,vymax)
        # test various limits on timestep (don't want fluid to move more than one cell per step)
        dtTest_speed =  vel_limiter * dx/vmax
        dtTest_acc   =  acc_limiter * sqrt(2.0*dx/self.external_acceleration(amax(self.dno)) )
        dtTest_diff  =  diff_limiter * dx*dx/(4.*self.D)
        # take the minimum of the criteria
        dtTest = min(dtTest_speed,dtTest_acc,dtTest_diff)
        # if we can increase the timestep, don't do so by more than a factor of 2. 
        if dtTest < dt:
            dt=dtTest
        if dtTest > dt:
            dt = min(dt*2, dtTest)
        # print some info
        print "... speed_lim dt=",dtTest_speed," accel_lim dt=",dtTest_acc," diff_lim dt=",dtTest_diff,"new dt =",dt
        return dt

    def make_monitoring_plots(self,velocity_vectors=False):
        """
        This routine makes a couple of simple plots. 
        
        One plots the velocity divergence (which should be close to zero if everything were *perfect*)
        The other plots density. We can optionally add velocity vectors to the density plot. 
        """
        #density plot code
        figure(figsize=(10,4))
        subplot(122)
        title("velocity divergence")
        pcolormesh(self.xx,self.yy,self.div, cmap='viridis_r')
        axis('equal')
        colorbar()
    
        subplot(121)
        title("density")
        pcolormesh(self.xx,self.yy,self.dno, cmap='jet')
        axis('equal')
        colorbar()
        if (velocity_vectors):
            stride = int(self.nx/8)
            quiver(self.xx[::stride,::stride],
                   self.yy[::stride,::stride],
                   self.vxo[::stride,::stride],
                   self.vyo[::stride,::stride])
        
        show()


    def evolve_fluid(self, vxlist,vylist,dnlist,
                        tmax=10.0, steps_max=1000, 
                        plot_every=10, save_every=100,
                        bctype="wrap",
                        dt_vel_limiter=0.5,dt_accel_limiter=0.25,dt_diff_limiter=0.5,
                        vel_vectors_in_monitor_plots=True):
        
        """
        This function evolves the fluid system from some initial conditions forward in time. 
        
        The code takes timesteps and can be configured to save or show outplot. 
        
        Some info is printed to the screen each timestep. 
        
        Parameters:
        vxlist,vylist,dnlist = These are user-supplied empty lists which will be filled with output of vx,vy,density
        
        tmax = max time to evolve the system to. 
        
        steps_max = maximum allowed timesteps
        
        plot_every = show a monitoring plot every X timesteps to screen
        
        save_every = save data (by appending it to the lists inputted) every X timesteps. 
                     This data can be used for plotting later. 
        
        dt_vel_limiter, dt_accel_limiter, dt_diff_limiter = safety factors below the maximum timestep (must be between 0-1)
        
        vel_vectors_in_monitor_plots = True/False show velocity arrows in plots printed to screen. 
        
        """
        
            
        start = time.time()
        # loop over timesteps
        for i in range(steps_max):
            
            print ""
            # pick a new timestep based on fluid conditions
            self.dt = self.choose_new_dt(vel_limiter=dt_vel_limiter,
                                         acc_limiter=dt_accel_limiter,
                                         diff_limiter=dt_diff_limiter)
            print "Taking step", i," with dt=",self.dt,"time=",self.current_time
            # call update to fluid motion by one step
            self.update_fluid_motion_step(flowbc=bctype)

            # update the current time
            self.current_time += self.dt
            # make a plot if requested
            if (i % plot_every == 0):
                self.make_monitoring_plots(velocity_vectors=vel_vectors_in_monitor_plots)
                    
            # save velocities and density every save_every steps
            if (i % save_every == 0):
                vxlist.append( copy(self.vxo) )
                vylist.append( copy(self.vyo) )
                dnlist.append( copy(self.dno) )
                
                    
                    
            # check if we've reached tmax
            if (self.current_time >= tmax):
                break
                
                
        
        end = time.time()
        if (self.current_time >= tmax):
            print "------------------------------------------------------"
            print " Stopping because t>tmax!!!"
            print " Evolved the fluid system to time of  ", self.current_time       
            print " taking ", i, " steps"
            print " computation took",end-start, " seconds"
            print "------------------------------------------------------"
        else:    
            print "------------------------------------------------------"
            print " Stopping because nsteps>steps_max!!!"
            print " Evolved the fluid system to time of  ", self.current_time       
            print " taking ", i, " steps"
            print " computation took",end-start, " seconds"
            print "------------------------------------------------------"
            
        self.make_monitoring_plots(velocity_vectors=vel_vectors_in_monitor_plots)
        return None
        

