# hmm_app.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) 
#

import os
import random
import tkinter as tk
import time

from collections import namedtuple
Agent = namedtuple("Agent", "loc")

class Tracker(object):
    def __init__(self, agent, landmarks, noise=0.1):
        self.landmarks = landmarks
        self.agent = agent
        self.noise = noise
    def sense(self):
        observation = []
        for sensor_x, sensor_y in self.landmarks:
            noise = random.gauss(0, self.noise)
            observation.append(
                ((self.agent.loc[0]-sensor_x)**2 + (self.agent.loc[1]-sensor_y)**2)**0.5 + noise
            )
        return observation


class App(tk.Frame):

    GRID_SIZE = (128, 128)
    LANDMARK_COLOR = "#000000"
    AGENT_COLOR = (.6, 0, .3, .5, .8)
    CLEMSON_LOGO = None
    SENSOR_NOISE = 0.2

    def __init__(self, num_agents, algs, master=None):
        super().__init__(master)
        self.algs = algs
        self.num_agents = min(num_agents, len(self.AGENT_COLOR))
        self.master.title("Hidden Markov Models -- CPSC 4420/6420 Clemson University")
        self.master.geometry("800x600")
        self.master.resizable(False, False)

        self.CLEMSON_LOGO = tk.PhotoImage(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "clemson.png"))
        self.AGENT_COLOR = self.AGENT_COLOR[0:self.num_agents]

        self.canvas = tk.Canvas(self.master, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4,
            sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=28
        )
        self.bt_new = tk.Button(self.master, text="New", command=self.new_game)
        self.alg_var = tk.StringVar(self.master)
        self.alg_var.set(next(iter(self.algs.keys())))
        self.current_alg = self.alg_var.get()
        self.listbox_alg = tk.OptionMenu(self.master, self.alg_var, *self.algs.keys(), command=self.change_alg)
        self.bt_run = tk.Button(self.master, text="Run", command=self.switch_autorun)

        self.bt_new.grid(row=1, column=0,
            sticky=tk.W, padx=10, pady=(0, 10)
        )
        self.listbox_alg.grid(row=1, column=2,
            sticky=tk.E, padx=10, pady=(0, 10)
        )
        self.bt_run.grid(row=1, column=3,
            sticky=tk.E, padx=10, pady=(0, 10)
        )
        self.autorun = False

        self.master.columnconfigure(0, weight=0)
        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(2, weight=1)
        self.master.columnconfigure(3, weight=0)
        self.master.rowconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=0)

        self.update_trigger = None
        self.canvas.bind("<Configure>", lambda event: self.new_game())
        
        self.start_time = 0

    def change_alg(self, selection):
        if self.current_alg == selection:
            return
        self.current_alg = selection
        self.new_game()

    def new_game(self):
        if self.update_trigger is not None:
            self.canvas.after_cancel(self.update_trigger)
            self.update_trigger = None
        self.canvas.delete("all")

        resolution = (1./self.GRID_SIZE[1], 1./self.GRID_SIZE[0])
        min_resolution = (resolution[0]*0.5, resolution[1]*0.5)
        max_resolution = (1.-min_resolution[0], 1.-min_resolution[1])

        
        new_size = min(self.canvas.winfo_width()/self.GRID_SIZE[1]*1.25, self.canvas.winfo_height()/self.GRID_SIZE[0]*1.25)
        sx = int(self.CLEMSON_LOGO.width()/new_size)
        sy = int(self.CLEMSON_LOGO.height()/new_size)
        self.clemson_logo = self.CLEMSON_LOGO.subsample(sx, sy)

        self.grid = [[None]*self.GRID_SIZE[1] for _ in range(self.GRID_SIZE[0])]

        self.landmarks = (
            (min_resolution[0], min_resolution[1]),
            (max_resolution[0], min_resolution[1]),
            (max_resolution[0], max_resolution[1]),
            (min_resolution[0], max_resolution[1]),
        )

        self.sensors = [
            Tracker(
                Agent(loc=[
                    random.uniform(min_resolution[0], max_resolution[0]),
                    random.uniform(min_resolution[1], max_resolution[1])
                ]),
                self.landmarks)
            for _ in self.AGENT_COLOR
        ]

        self.inferencers = [self.algs[self.alg_var.get()](self.GRID_SIZE[0], self.GRID_SIZE[1]) for _ in self.AGENT_COLOR]

            
        self.draw_grid()

        def flush(iteration):
            if (100 == iteration):
                print(f"Time for 100 iterations in seconds: {time.perf_counter() - self.start_time:.4f}")
            self.canvas.delete("belief")
            self.canvas.delete("agent")
            beliefs = []
            for infer, s, color in zip(self.inferencers, self.sensors, self.AGENT_COLOR):
                infer.observe(s.sense(), self.landmarks)
                
                if self.autorun:
                    dx = int(random.uniform(0, 3)) - 1
                    dy = int(random.uniform(0, 3)) - 1

                    s.agent.loc[0] += resolution[0] * dx
                    s.agent.loc[1] += resolution[1] * dy
                    s.agent.loc[0] = max(min_resolution[0], min(max_resolution[0], s.agent.loc[0]))
                    s.agent.loc[1] = max(min_resolution[1], min(max_resolution[1], s.agent.loc[1]))

                    infer.timeUpdate()
                    iteration += 1
                beliefs.append(infer.belief)

            self.draw_belief(beliefs, self.AGENT_COLOR)
            self.draw_landmarks()
            self.draw_agents()
            self.update_trigger = self.canvas.after(1, flush, iteration)

        self.update_trigger = self.canvas.after(1, flush, 0)

    def switch_autorun(self):
        self.autorun = not self.autorun
        if self.autorun:
            self.start_time = time.perf_counter()
            self.bt_run["text"] = "Pause"
        else:
            self.bt_run["text"] = "Run"

    def draw_grid(self, event=None):
        self.canvas.delete("grid_line")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        c_interval = w / self.GRID_SIZE[1]
        r_interval = h / self.GRID_SIZE[0]
        for c in range(self.GRID_SIZE[1]):
            self.canvas.create_line([(c_interval*c, 0), (c_interval*c, r_interval*h)], tag="grid_line")
        for r in range(self.GRID_SIZE[0]):
            self.canvas.create_line([(0, r_interval*r), (c_interval*w, r_interval*r)], tag="grid_line")

    def draw_agents(self):
        self.canvas.delete("agent")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        c_interval = w / self.GRID_SIZE[1] * 0.65
        r_interval = h / self.GRID_SIZE[0] * 0.65
        for s, color in zip(self.sensors, self.AGENT_COLOR):
            x, y = s.agent.loc
            # self.canvas.create_oval(
            #     x*w-c_interval, (1-y)*h-r_interval,
            #     x*w+c_interval, (1-y)*h+r_interval,
            #     fill=self.rgb2hex(*self.hsv2rgb(color, 1., .66)), width=1,
            #     tag="agent"
            # )
            self.canvas.create_image(
                x*w, (1-y)*h,
                anchor=tk.CENTER, image=self.clemson_logo,
                tag="agent"
            )
    
    def draw_landmarks(self):
        self.canvas.delete("landmark")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        c_interval = w / self.GRID_SIZE[1] * 0.5
        r_interval = h / self.GRID_SIZE[0] * 0.5
        for x, y in self.landmarks:
            self.canvas.create_rectangle(
                x*w-c_interval, (1-y)*h-r_interval,
                x*w+c_interval, (1-y)*h+r_interval,
                fill=self.LANDMARK_COLOR, width=1,
                tag="landmark"
            )

    def draw_belief(self, beliefs, agent_colors):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        c_interval = w / self.GRID_SIZE[1]
        r_interval = h / self.GRID_SIZE[0]
        avg_belief = 1 / (self.GRID_SIZE[0]*self.GRID_SIZE[1])
        color = [[None]*self.GRID_SIZE[1] for _ in range(self.GRID_SIZE[0])]
        for belief, agent_color in zip(beliefs, agent_colors):
            assert(len(belief) == self.GRID_SIZE[0] and len(belief[0]) == self.GRID_SIZE[1])
            sum_alpha = 0.
            n = 0
            for r in range(len(belief)):
                for c in range(len(belief[0])):
                    alpha = belief[r][c] - avg_belief
                    if alpha > 0:
                        sum_alpha += alpha
                        n+=1
            if sum_alpha:
                alpha_scale = max(1, n/4)/sum_alpha
            else: break
            for r in range(len(belief)):
                for c in range(len(belief[0])):
                    alpha = belief[r][c] - avg_belief
                    if alpha > 0:
                        alpha *= alpha_scale
                        alpha = min(1., alpha)
                        if color[r][c]:
                            color[r][c] = tuple(min(1, (1-alpha)*e+alpha*e_) for e, e_ in zip(color[r][c], self.hsv2rgb(agent_color, alpha, 1.)))
                        else:
                            color[r][c] = self.hsv2rgb(agent_color, alpha, 1.)
        for r in range(len(belief)):
            for c in range(len(belief[0])):
                if color[r][c]:
                    self.canvas.create_rectangle(
                        c_interval*c, r_interval*r,
                        c_interval*(c+1), r_interval*(r+1),
                        fill=self.rgb2hex(*color[r][c]),
                        tag="belief"
                    )
    
    @staticmethod
    def hsv2rgb(h, s, v):
        if s > 0:
            i = int(h*6.)
            f = h*6. - i
            p = v*(1-s)
            q = v*(1-s*f)
            t = v*(1-s*(1-f)) 
            if i == 1:
                return q, v, p
            elif i == 2:
                return p, v, t
            elif i == 3:
                return p, q, v
            elif i == 4:
                return t, p, v
            elif i == 5:
                return v, p, q
            else:
                return v, t, p
        return v, v, v
    
    @staticmethod
    def rgb2hex(r, g, b):
        return "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))
